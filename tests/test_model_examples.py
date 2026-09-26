"""Offline regressions for real Torch export, Transformer causality, and metrics."""

from collections.abc import Callable, Iterator, Mapping
from typing import Literal

import numpy as np
import onnx
import pytest
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import CliApp
from torch import nn

from examples._shared.schema import QuantizationMethod
from examples._shared.training.data import text_windows, wine
from examples._shared.training.measurement import Totals
from examples._shared.training.schema import Config, TaskKind
from examples._shared.training.training import eager
from examples.projection import main as projection_example
from examples.projection.model import example_model
from examples.transformer import main as transformer_example
from examples.transformer.model import CharacterTransformer
from examples.windowed_transformer import main as windowed_example
from examples.windowed_transformer.model import WindowedCharacterTransformer
from examples.wine_mlp import main as wine_example
from examples.wine_mlp.model import WineMLP
from qraft.algorithms import Algorithm
from qraft.backends.onnx import describe
from qraft.backends.onnx.pipeline import quantize
from qraft.domain import FloatArray, InputArray
from qraft.rules import Rules
from qraft.runtime import OnnxEvaluator
from tests.outcomes import expect_error, expect_ok

type ModelKind = Literal[
    TaskKind.TRANSFORMER, TaskKind.WINDOWED, TaskKind.WINE, "projection"
]


class ExportCase(BaseModel):
    """Pair a real Torch architecture with its serialized graph and native input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: nn.Module = Field()
    graph: onnx.ModelProto = Field()
    inputs: InputArray = Field()
    kind: ModelKind = Field()
    input_name: str = Field()
    output_name: str = Field()


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> None:
    """Keep CPU tests reproducible and avoid native thread oversubscription."""
    torch.set_num_threads(1)


@pytest.fixture(
    scope="module",
    params=[TaskKind.TRANSFORMER, TaskKind.WINDOWED, TaskKind.WINE, "projection"],
)
def export_case(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ExportCase:
    """Export the available Torch architectures offline and the ONNX projection."""
    kind: ModelKind = request.param
    torch.manual_seed(7)
    builders: dict[TaskKind, Callable[[int], nn.Module]] = {
        TaskKind.TRANSFORMER: CharacterTransformer,
        TaskKind.WINDOWED: WindowedCharacterTransformer,
    }
    match kind:
        case "projection":
            graph = example_model()
            model = nn.Linear(8, 4, bias=False).eval()
            weight = np.asarray(
                onnx.numpy_helper.to_array(graph.graph.initializer[0]), dtype=np.float32
            )
            with torch.no_grad():
                model.weight.copy_(torch.from_numpy(weight.T.copy()))
            return ExportCase(
                model=model,
                graph=graph,
                inputs=np.ones((1, 8), dtype=np.float32),
                kind=kind,
                input_name="x",
                output_name="y",
            )
        case TaskKind.WINE:
            model = WineMLP().eval()
            inputs: InputArray = np.ones((1, 13), dtype=np.float32)
        case TaskKind() as task:
            model = builders[task](8).eval()
            inputs = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    destination = tmp_path_factory.mktemp(str(kind)) / "fp32.onnx"
    torch.onnx.export(
        model,
        (torch.from_numpy(inputs),),
        str(destination),
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    graph = onnx.load(destination)
    return ExportCase(
        model=model,
        graph=graph,
        inputs=inputs,
        kind=kind,
        input_name="input",
        output_name="logits",
    )


def example_stages(
    kind: ModelKind, method: QuantizationMethod
) -> tuple[Rules[Algorithm], ...]:
    """Exercise the configurations written in each example rather than a test recipe."""
    match kind:
        case TaskKind.TRANSFORMER:
            return transformer_example.recipes(
                transformer_example.Config(histogram_bins=64)
            )[method]
        case TaskKind.WINDOWED:
            return windowed_example.recipes(windowed_example.Config(histogram_bins=64))[
                method
            ]
        case TaskKind.WINE:
            return wine_example.recipes(wine_example.Config(histogram_bins=64))[method]
        case "projection":
            return projection_example.recipes(
                projection_example.Config(histogram_bins=64)
            )[method].stages()


@pytest.mark.parametrize("method", list(QuantizationMethod))
def test_torch_export_quantization_roundtrip(
    export_case: ExportCase, method: QuantizationMethod
) -> None:
    """Quantize independent recipes while preserving shape and source ownership."""
    before = export_case.graph.SerializeToString()

    def source() -> Iterator[Mapping[str, InputArray]]:
        """Replay native feature/token inputs without conversion to another dtype."""
        yield {export_case.input_name: export_case.inputs}

    result = expect_ok(
        quantize(
            export_case.graph,
            source,
            example_stages(export_case.kind, method),
            histogram_bins=64,
        )
    )
    lowered = result.model
    onnx.checker.check_model(lowered)
    actual = expect_ok(
        OnnxEvaluator(lowered).run(
            {export_case.input_name: export_case.inputs}, (export_case.output_name,)
        )
    )[export_case.output_name]
    expected = eager(export_case.model, export_case.inputs)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert float(np.mean((actual - expected) ** 2)) < 0.1
    assert export_case.graph.SerializeToString() == before


@pytest.mark.parametrize(
    "builder", [CharacterTransformer, WindowedCharacterTransformer]
)
def test_transformer_causality(builder: Callable[[int], nn.Module]) -> None:
    """Future token changes must not change earlier next-token predictions."""
    model = builder(8).eval()
    first = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    second = first.copy()
    second[:, 4:] = 0
    np.testing.assert_allclose(
        eager(model, first)[:, :4], eager(model, second)[:, :4], atol=1e-06
    )


def test_dynamic_attention_edges_remain_float(export_case: ExportCase) -> None:
    """Constant-weight recipes cannot pretend to quantize dynamic QK/AV products."""
    graph = expect_ok(describe(export_case.graph))
    dynamic = [
        node
        for node in graph.nodes
        if node.op == "MatMul" and node.inputs[1] not in graph.weights
    ]
    if export_case.kind in (TaskKind.TRANSFORMER, TaskKind.WINDOWED):
        assert len(dynamic) >= 4
    for node in dynamic:
        assert not transformer_example.ConstantWeights().matches(node, graph)


def test_wine_split_and_normalization() -> None:
    """Prove training-only normalization and disjoint real dataset identities."""
    data = wine(Config())
    train, calibration, evaluation = (
        set(split.identities)
        for split in (data.train, data.calibration, data.evaluation)
    )
    assert not (train & calibration or train & evaluation or calibration & evaluation)
    assert len(train | calibration | evaluation) == 178
    np.testing.assert_allclose(data.train.inputs.mean(axis=0), 0, atol=1e-05)
    np.testing.assert_allclose(data.train.inputs.std(axis=0), 1, atol=1e-05)
    assert data.calibration.identities == wine(Config()).calibration.identities


def test_text_window_boundaries() -> None:
    """Input and shifted targets must stay inside their assigned source span."""
    tokens = np.arange(200, dtype=np.int64)
    split = expect_ok(text_windows(tokens, 100, 200, 8, 8))
    assert int(split.inputs.min()) >= 100
    assert int(split.targets.max()) < 200
    np.testing.assert_array_equal(split.targets, split.inputs + 1)
    expect_error(text_windows(tokens, 100, 200, 100, 8), "available")


def test_token_metrics() -> None:
    """Count each token separately and verify exact entropy on uniform logits."""
    logits: FloatArray = np.zeros((1, 2, 2), dtype=np.float32)
    totals = Totals()
    totals.update(logits, logits, logits, np.asarray([[0, 1]], dtype=np.int64))
    scores = totals.finish()
    assert scores.observations == 2
    assert scores.accuracy == 0.5
    assert scores.cross_entropy == pytest.approx(np.log(2))
    assert scores.perplexity == pytest.approx(2)
    assert scores.mse_vs_onnx == 0


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ("percentile", [QuantizationMethod.PERCENTILE]),
        (
            "minmax,percentile",
            [QuantizationMethod.MINMAX, QuantizationMethod.PERCENTILE],
        ),
    ],
)
def test_typed_task_cli(selection: str, expected: list[QuantizationMethod]) -> None:
    """Select methods while each example retains its explicit model identity."""
    config = CliApp.run(
        transformer_example.Config, cli_args=["--methods", selection, "--quiet"]
    )
    assert config.tasks == (TaskKind.TRANSFORMER,)
    assert config.methods == expected
    assert config.quiet


@pytest.mark.parametrize("config_type", [Config])
def test_cli_configuration_roundtrip(config_type: type[Config]) -> None:
    """Saved reports must accept both the CLI alias and serialized Python field name."""
    config = CliApp.run(config_type, cli_args=["--json", "--quiet"])
    restored = config_type.model_validate_json(config.model_dump_json())
    assert restored == config
    assert restored.json_output
