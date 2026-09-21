"""Shared trajectory metrics for the case-study benchmarks."""

import numpy as np

# Loss floor 10**-C, below any loss reached, so the area stays non-negative.
LOG_LEARNING_AREA_REFERENCE = 10.0


def log_learning_area(
    train_loss_curve, *, reference: float = LOG_LEARNING_AREA_REFERENCE
) -> float:
    """Area under one run's log10 training-loss curve: ``mean(reference +
    log10(L(t)))`` over finite positive steps."""

    curve = np.asarray(train_loss_curve, dtype=float)
    usable = curve[np.isfinite(curve) & (curve > 0.0)]
    if usable.size == 0:
        return float("nan")
    return float(reference + np.mean(np.log10(usable)))


def seed_statistics(values) -> dict[str, float | None]:
    """min / mean / max / population std and 25th / 50th / 75th percentiles over the
    finite values; the percentiles interpolate linearly, as in the loss figures."""

    array = np.asarray([value for value in values if value is not None], dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return dict.fromkeys(("min", "mean", "max", "std", "q25", "median", "q75"))
    q25, median, q75 = np.percentile(finite, (25, 50, 75))
    return {
        "min": float(np.min(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "std": float(np.std(finite)),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
    }
