"""Shared plumbing for the per-case SGD wall-clock timing baseline."""


from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


# Outside results/, so run.sh's wipe of results/ never touches the baseline.
SGD_TIMING_DIRNAME = "sgd_timing"
SGD_TIMING_JSON_NAME = "sgd_timing.json"
_CACHE_DIRNAME = "cache"


def timing_dir(case_folder: str | Path, dirname: str = SGD_TIMING_DIRNAME) -> Path:
    """The folder holding everything a case's SGD baseline produces."""

    return Path(case_folder) / dirname


def timing_json(case_folder: str | Path, dirname: str = SGD_TIMING_DIRNAME) -> Path:
    """The aggregated per-regime timings the paper reads."""

    return timing_dir(case_folder, dirname) / SGD_TIMING_JSON_NAME


def cache_dir(case_folder: str | Path, dirname: str = SGD_TIMING_DIRNAME) -> Path:
    """Where the per-``(regime, seed)`` readings are cached."""

    return timing_dir(case_folder, dirname) / _CACHE_DIRNAME


_DEFAULT_SGD_SEED_COUNT = 10
_SGD_SEED_COUNT_ENV = "SGD_TIMING_SEEDS"


def sgd_seed_count() -> int:
    """Number of SGD timing seeds, from ``$SGD_TIMING_SEEDS`` (default 10)."""

    raw = os.environ.get(_SGD_SEED_COUNT_ENV)
    if raw is None or raw.strip() == "":
        return _DEFAULT_SGD_SEED_COUNT
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(
            f"{_SGD_SEED_COUNT_ENV} must be a positive integer, got {raw!r}"
        ) from None
    if count < 1:
        raise ValueError(f"{_SGD_SEED_COUNT_ENV} must be at least 1, got {count}")
    return count


def sgd_seeds() -> tuple[int, ...]:
    """The SGD timing seeds, ``(0, 1, ..., sgd_seed_count() - 1)``."""

    return tuple(range(sgd_seed_count()))


# Default count only; call sgd_seeds() to honor $SGD_TIMING_SEEDS.
SGD_SEEDS: tuple[int, ...] = tuple(range(_DEFAULT_SGD_SEED_COUNT))

_NOTE = (
    "plain SGD (w <- w - alpha * g, no momentum); wall-clock timing "
    "normalizer for the training-time tables only -- not a benchmarked "
    "method (absent from optimizer_config.METHODS)"
)


def aggregate(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """min / mean / max / population-std / count over the finite entries."""

    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    mean = sum(finite) / len(finite)
    std = (sum((x - mean) ** 2 for x in finite) / len(finite)) ** 0.5
    return {
        "mean": mean,
        "std": std,
        "min": min(finite),
        "max": max(finite),
        "n": len(finite),
    }


def regime_payload(
    *,
    per_seed_seconds: Sequence[float],
    per_seed_final_loss: Sequence[float | None],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One regime's entry for ``sgd_timing/sgd_timing.json``."""

    diverged = sum(
        1
        for value in per_seed_final_loss
        if value is None or not math.isfinite(float(value))
    )
    return {
        **(extra or {}),
        "training_seconds": aggregate(per_seed_seconds),
        "per_seed_training_seconds": [float(value) for value in per_seed_seconds],
        "final_train_loss": aggregate(per_seed_final_loss),
        "diverged_seeds": diverged,
    }


def write_timing_json(
    path: str | Path,
    *,
    case: str,
    regimes: dict[str, Any],
    seeds: Sequence[int] | None = None,
) -> Path:
    """ """

    seeds = sgd_seeds() if seeds is None else seeds
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": case,
        "optimizer": "sgd",
        "note": _NOTE,
        "seeds": list(seeds),
        "regimes": regimes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cached_child_run(
    *,
    script: str | Path,
    child_args: Sequence[Any],
    cache_file: str | Path,
) -> dict[str, Any]:
    """Return one ``(regime, seed)`` reading, running it in a fresh process if it is
    not already cached."""

    cache_file = Path(cache_file)
    if not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--child",
                *[str(argument) for argument in child_args],
                str(cache_file),
            ],
            check=True,
        )
    return json.loads(cache_file.read_text())


def run_subprocess_regimes(
    *,
    script: str | Path,
    case: str,
    regimes: Sequence[tuple[str, dict[str, Any]]],
    cache_dir: str | Path,
    out_json: str | Path,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Driver for the multi-regime cases: one subprocess per ``(regime, seed)``."""

    seeds = sgd_seeds() if seeds is None else seeds
    cache_dir = Path(cache_dir)
    regime_entries: dict[str, Any] = {}
    for regime_key, meta in regimes:
        seconds: list[float] = []
        losses: list[float | None] = []
        for seed in seeds:
            reading = cached_child_run(
                script=script,
                child_args=[regime_key, seed, json.dumps(meta, sort_keys=True)],
                cache_file=cache_dir / f"{regime_key}__seed{seed}.json",
            )
            seconds.append(reading["training_seconds"])
            losses.append(reading.get("final_train_loss"))
        regime_entries[regime_key] = regime_payload(
            per_seed_seconds=seconds,
            per_seed_final_loss=losses,
            extra={key: value for key, value in meta.items() if key != "overrides"},
        )
        write_timing_json(out_json, case=case, regimes=regime_entries, seeds=seeds)
    return Path(out_json)


def run_inprocess_regimes(
    *,
    case: str,
    regimes: Sequence[tuple[str, Callable[[int], tuple[float, float | None]], dict[str, Any] | None]],
    out_json: str | Path,
    cache_dir: str | Path,
    seeds: Sequence[int] | None = None,
) -> Path:
    """Driver for the in-process cases (PINN / CIFAR-10 / U-Net): loop the seeds of
    every regime in one process."""

    seeds = sgd_seeds() if seeds is None else seeds
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    regime_entries: dict[str, Any] = {}
    for regime_key, seconds_and_loss, meta in regimes:
        seconds: list[float] = []
        losses: list[float | None] = []
        for seed in seeds:
            cache_file = cache_dir / f"{regime_key}__seed{seed}.json"
            if cache_file.exists():
                reading = json.loads(cache_file.read_text())
            else:
                training_seconds, final_loss = seconds_and_loss(seed)
                reading = {
                    "training_seconds": float(training_seconds),
                    "final_train_loss": (
                        None if final_loss is None or not math.isfinite(float(final_loss))
                        else float(final_loss)
                    ),
                    "regime": regime_key,
                    "seed": seed,
                }
                cache_file.write_text(json.dumps(reading), encoding="utf-8")
            seconds.append(reading["training_seconds"])
            losses.append(reading.get("final_train_loss"))
        regime_entries[regime_key] = regime_payload(
            per_seed_seconds=seconds, per_seed_final_loss=losses, extra=meta or {}
        )
        write_timing_json(out_json, case=case, regimes=regime_entries, seeds=seeds)
    return Path(out_json)


def run_inprocess_single_regime(
    *,
    case: str,
    regime_key: str,
    seconds_and_loss: Callable[[int], tuple[float, float | None]],
    out_json: str | Path,
    cache_dir: str | Path,
    meta: dict[str, Any] | None = None,
    seeds: Sequence[int] | None = None,
) -> Path:
    """``run_inprocess_regimes`` for a case with exactly one regime."""

    return run_inprocess_regimes(
        case=case,
        regimes=[(regime_key, seconds_and_loss, meta)],
        out_json=out_json,
        cache_dir=cache_dir,
        seeds=seeds,
    )
