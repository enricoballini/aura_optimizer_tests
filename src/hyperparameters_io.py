"""Plain-text read/write helpers for one case's fixed-configuration hyperparameters."""


import ast
from pathlib import Path
from typing import Any


def write_hyperparameters_txt(path: Path, values: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key} = {value!r}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_hyperparameters_txt(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameters file does not exist: {path}") from None

    values: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key, separator, raw_value = line.partition("=")
        if not separator:
            raise ValueError(f"Malformed line in {path}: {line!r}")
        values[key.strip()] = ast.literal_eval(raw_value.strip())
    return values
