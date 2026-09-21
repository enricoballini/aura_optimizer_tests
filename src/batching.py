"""Shared minibatch iteration for every optimizer-comparison case."""


import jax


def batches_per_epoch(train_size: int, batch_size: int) -> int:
    """Number of minibatch updates that ``epoch_batches`` yields per epoch."""

    if train_size < 1:
        raise ValueError(f"train_size must be at least 1, got {train_size!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size!r}")
    return max(1, train_size // min(train_size, batch_size))


def epoch_batches(key, train_size: int, batch_size: int, steps: int):
    """Yield ``steps`` index arrays as successive shuffled passes over
    ``range(train_size)``."""

    effective_batch_size = min(train_size, batch_size)
    usable = batches_per_epoch(train_size, batch_size) * effective_batch_size

    permutation = None
    position = usable  # forces a shuffle before the first batch
    for _ in range(steps):
        if position >= usable:
            key, subkey = jax.random.split(key)
            permutation = jax.random.permutation(subkey, train_size)
            position = 0
        yield permutation[position:position + effective_batch_size]
        position += effective_batch_size
