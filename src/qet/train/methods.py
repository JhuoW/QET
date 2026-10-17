"""Training configurations as edge weights (SPEC §8): the six core configurations and the ablations.

Every configuration runs the same training loop (`qet.train.trainer`); they differ only in the renderer, the edge
weights and the start states. Configuration 1 differs from configuration 2 only in its renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

from qet.core.render import VARIANTS
from qet.core.weights import METHODS as WEIGHT_RULES

PASS1_RULES = ("frontier", "qet", "sampled_paths")  # weights that depend on the model's own probabilities


@dataclass(frozen=True)
class Method:
    config_id: str
    name: str
    renderer: str  # quotient | chronological | native_multiturn (SPEC §5.7)
    weights: str  # a weight rule of qet.core.weights: path_sft | uniform_aug | frontier | qet | sampled_paths
    lam: float = 0.0  # lambda of the start distribution mu; 0 starts at the empty state only
    order: str | None = None  # path_sft: "demo" or "canonical" (ablation a2)
    paths: int | None = None  # sampled_paths: K, the demo order plus K - 1 random orders (D-011)
    force_sublattice: bool = False  # a4: the sub-lattice on every graph, not only above S_max (D-023)

    def __post_init__(self) -> None:
        if self.renderer not in VARIANTS:
            raise ValueError(f"unknown renderer {self.renderer!r}")
        if self.weights not in WEIGHT_RULES:
            raise ValueError(f"unknown weight rule {self.weights!r}")
        if (self.weights == "path_sft") != (self.order is not None) or self.order not in (None, "demo", "canonical"):
            raise ValueError("path_sft takes order 'demo' or 'canonical'; other rules take none")
        if (self.weights == "sampled_paths") != (self.paths is not None):
            raise ValueError("sampled_paths takes the number of paths K; other rules take none")
        if self.renderer != "quotient" and (self.weights, self.order) != ("path_sft", "demo"):
            raise ValueError("order-sensitive renderers are trained on the demo order only")
        if not 0.0 <= self.lam < 1.0:
            raise ValueError(f"lambda must lie in [0, 1), got {self.lam}")

    @property
    def pass1(self) -> bool:
        """Whether the weights need pass 1 (the model's probabilities before the update)."""
        return self.weights in PASS1_RULES

    @property
    def lattice(self) -> str:
        """The states trained on: the demo (or canonical) path, the sampled paths, or the lattice."""
        return {"path_sft": "path", "sampled_paths": "paths"}.get(self.weights, "lattice")


METHODS: dict[str, Method] = {
    m.config_id: m
    for m in (
        Method("1", "chronological_sft", "chronological", "path_sft", order="demo"),
        Method("2", "demo_order_sft", "quotient", "path_sft", order="demo"),
        Method("3", "uniform_augmentation", "quotient", "uniform_aug", lam=0.5),
        Method("4", "frontier_supervision", "quotient", "frontier", lam=0.5),
        Method("5", "qet_lambda0", "quotient", "qet", lam=0.0),
        Method("6", "qet", "quotient", "qet", lam=0.5),
        Method("a1", "native_multiturn_sft", "native_multiturn", "path_sft", order="demo"),
        Method("a2", "canonical_order_sft", "quotient", "path_sft", order="canonical"),
        Method("a3", "sampled_order_marginalization", "quotient", "sampled_paths", paths=4),
        Method("a4", "sublattice_qet", "quotient", "qet", lam=0.5, force_sublattice=True),
    )
}


def method(config_id: str) -> Method:
    if config_id not in METHODS:
        raise ValueError(f"unknown configuration {config_id!r}; expected one of {sorted(METHODS)}")
    return METHODS[config_id]
