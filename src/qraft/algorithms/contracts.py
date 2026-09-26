"""Backend-independent inputs and outputs for quantization algorithms."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qraft.calibration import HistogramStats, MinMaxStats, Requirement
from qraft.domain import Graph, Node
from qraft.plan import QuantizationPlan
from qraft.result import QraftError, Result


class Needs(BaseModel):
    """Declare reusable range and histogram requirements for an algorithm."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    ranges: tuple[Requirement, ...] = Field(default=())
    histograms: tuple[Requirement, ...] = Field(default=())


class Statistics(BaseModel):
    """Own read-only statistics mappings for exactly one graph revision."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    ranges: Mapping[Requirement, MinMaxStats] = Field()
    histograms: Mapping[Requirement, HistogramStats] = Field()

    @field_validator("ranges")
    @classmethod
    def freeze_ranges(
        cls, value: Mapping[Requirement, MinMaxStats]
    ) -> Mapping[Requirement, MinMaxStats]:
        """Detach the range mapping from the caller's mutable collection."""
        return MappingProxyType(dict(value))

    @field_validator("histograms")
    @classmethod
    def freeze_histograms(
        cls, value: Mapping[Requirement, HistogramStats]
    ) -> Mapping[Requirement, HistogramStats]:
        """Detach the histogram mapping from the caller's mutable collection."""
        return MappingProxyType(dict(value))


@runtime_checkable
class Algorithm(Protocol):
    """Produce decisions without execution, I/O, or mutation of input records.

    Implementations return expected failures through Result. Unexpected plugin
    exceptions propagate; the protocol cannot enforce numerical correctness or purity.
    """

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QraftError]:
        """Declare statistics needed for planning or reject an unsupported layout."""
        ...

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QraftError]:
        """Return a plan from supplied statistics without executing the graph."""
        ...
