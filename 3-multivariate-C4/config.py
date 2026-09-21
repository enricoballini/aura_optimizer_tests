"""Configuration objects and faithful reference hyperparameters."""


from dataclasses import asdict, dataclass
from typing import Any

from src import batching
from src import optimizer_config
import dataset


def steps_per_epoch(train_size: int, minibatch_size: int) -> int:
    """ """

    return batching.batches_per_epoch(train_size, minibatch_size)


@dataclass(frozen=True)
class ExperimentConfig:
    """ """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    steps: int = 12000
    train_size: int = 1296
    minibatch_size: int = 128
    # Single architecture: secondary_* mirrors it so the two-architecture code runs once.
    hidden_width: int = 128
    hidden_layers: int = 5
    secondary_hidden_width: int = 128
    secondary_hidden_layers: int = 5
    learning_rate: float = 1e-4
    small_network_learning_rate_multiplier: float = 0.1
    # Unused by this case, which instead samples dataset.C4_DOMAIN_MIN/MAX.
    domain_half_width: float = 1.0
    initialization_beta: float = 0.5
    seed: int = 0
    precision: str = "32"
    test_every: int = 100
    max_workers: int = 5

    def validate(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        for seed in self.seeds:
            if not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
                raise ValueError(
                    "seeds must contain integers between 0 and 2**32 - 1, "
                    f"got {seed!r}"
                )
        integer_fields = {
            "steps": self.steps,
            "train_size": self.train_size,
            "minibatch_size": self.minibatch_size,
            "hidden_width": self.hidden_width,
            "hidden_layers": self.hidden_layers,
            "secondary_hidden_width": self.secondary_hidden_width,
            "secondary_hidden_layers": self.secondary_hidden_layers,
            "test_every": self.test_every,
            "max_workers": self.max_workers,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 0xFFFFFFFF:
            raise ValueError(
                f"seed must be an integer between 0 and 2**32 - 1, got {self.seed!r}"
            )
        if self.test_size < 1:
            raise ValueError(
                "test_size must be at least 1 (train_size too small): "
                f"got train_size={self.train_size!r}, test_size={self.test_size!r}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.small_network_learning_rate_multiplier <= 0.0:
            raise ValueError("small_network_learning_rate_multiplier must be positive")
        if self.domain_half_width <= 0.0:
            raise ValueError("domain_half_width must be positive")
        if self.initialization_beta <= 0.0:
            raise ValueError("initialization_beta must be positive")
        if self.precision not in ("32", "64"):
            raise ValueError("precision must be '32' or '64'")

    @property
    def test_size(self) -> int:
        """Total number of held-out test points: 20% of ``train_size``."""

        return round(0.2 * self.train_size)

    @property
    def architecture(self) -> tuple[int, ...]:
        return (
            (dataset.TARGET_INPUT_WIDTH,)
            + (self.hidden_width,) * self.hidden_layers
            + (1,)
        )

    @property
    def architecture_settings(self) -> tuple[tuple[int, int], ...]:
        """Return the distinct ``(width, depth)`` benchmark architectures."""

        settings = (
            (self.hidden_width, self.hidden_layers),
            (self.secondary_hidden_width, self.secondary_hidden_layers),
        )
        return tuple(dict.fromkeys(settings))

    @property
    def architectures(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            (dataset.TARGET_INPUT_WIDTH,) + (width,) * layers + (1,)
            for width, layers in self.architecture_settings
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["test_size"] = self.test_size
        result["architecture"] = list(self.architecture)
        result["architectures"] = [list(item) for item in self.architectures]
        return result


def reference_hyperparameters() -> dict[str, dict[str, Any]]:
    """Return JSON-serializable optimizer settings used by the benchmark."""

    return {
        "adam": asdict(optimizer_config.ADAM_CONFIG),
        optimizer_config.METHOD_LBFGS: asdict(optimizer_config.LBFGS_CONFIG),
        optimizer_config.METHOD_RPROP: asdict(optimizer_config.RPROP_CONFIG),
        optimizer_config.METHOD_ADAM_AURA: asdict(optimizer_config.AURA_CONFIG),
        optimizer_config.METHOD_ASTRA: asdict(optimizer_config.ASTRA_CONFIG),
        optimizer_config.METHOD_ECLIPSE: asdict(optimizer_config.ECLIPSE_CONFIG),
        optimizer_config.METHOD_PULSAR: asdict(optimizer_config.PULSAR_CONFIG),
        optimizer_config.METHOD_AURA_LIGHT: asdict(optimizer_config.AURA_LIGHT_CONFIG),
        optimizer_config.METHOD_NADAMW: asdict(optimizer_config.NADAMW_CONFIG),
        optimizer_config.METHOD_NADAM: asdict(optimizer_config.NADAM_CONFIG),
        optimizer_config.METHOD_CvAMSGrad: asdict(optimizer_config.CVAMSGRAD_CONFIG),
        optimizer_config.METHOD_MUON: asdict(optimizer_config.MUON_CONFIG),
        optimizer_config.METHOD_MUON_AURA: asdict(optimizer_config.MUON_AURA_CONFIG),
        optimizer_config.METHOD_ADAM_AURA_SNR: asdict(optimizer_config.AURA_SNR_CONFIG),
        optimizer_config.METHOD_MUON_AURA_SNR: asdict(optimizer_config.MUON_AURA_SNR_CONFIG),
        optimizer_config.METHOD_HCSCGM: asdict(optimizer_config.HCSCGM_CONFIG),
    }
