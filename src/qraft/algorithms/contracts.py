"""Backend-independent inputs and outputs for quantization algorithms."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from returns.result import Result

from qraft.calibration import HistogramStats, MinMaxStats, Requirement
from qraft.domain import Graph, Node
from qraft.plan import QuantizationPlan
from qraft.result import QraftError


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

    @model_validator(mode="after")
    def freeze_mappings(self) -> Self:
        """Detach caller mappings so ordinary mutation cannot change planning inputs."""
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))
        object.__setattr__(self, "histograms", MappingProxyType(dict(self.histograms)))
        return self


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
