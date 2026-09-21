"""Generic read/write helpers for one method-run's full per-step history."""


import csv
from pathlib import Path

import numpy as np

_REAL_SELECTION_FIELDS = (
    ("selected_gamma_history", "selected_gamma"),
    ("selected_angle_history", "selected_angle"),
    ("selected_smoothed_angle_history", "selected_smoothed_angle"),
    ("selected_q_history", "selected_q"),
    ("selected_chi_history", "selected_chi"),
    ("selected_psi_history", "selected_psi"),
)


def write_method_history(
    path: Path,
    *,
    train_loss: np.ndarray,
    test_loss: np.ndarray,
    diagnostics: np.ndarray,
    train_accuracy: np.ndarray | None = None,
    test_accuracy: np.ndarray | None = None,
    weight_history: np.ndarray,
    selected_gamma_history: np.ndarray,
    selected_angle_history: np.ndarray | None = None,
    selected_smoothed_angle_history: np.ndarray | None = None,
    selected_q_history: np.ndarray | None = None,
    selected_chi_history: np.ndarray | None = None,
    selected_psi_history: np.ndarray | None = None,
) -> None:
    """Write one method-run's full per-step arrays as a single CSV, one row per step."""

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)
    diagnostics = np.asarray(diagnostics)
    weight_history = np.asarray(weight_history)
    steps = train_loss.shape[0]

    columns: list[str] = ["step", "train_loss", "test_loss"]
    data: list[np.ndarray] = [
        np.arange(steps),
        train_loss,
        test_loss,
    ]

    if train_accuracy is not None:
        columns.append("train_accuracy")
        data.append(np.asarray(train_accuracy))
    if test_accuracy is not None:
        columns.append("test_accuracy")
        data.append(np.asarray(test_accuracy))

    num_diagnostics = diagnostics.shape[1]
    for index in range(num_diagnostics):
        columns.append(f"diagnostic_{index}")
        data.append(diagnostics[:, index])

    is_complex_weights = np.iscomplexobj(weight_history)
    num_selections = weight_history.shape[1]
    for selection_index in range(num_selections):
        if is_complex_weights:
            columns.append(f"weight_re_{selection_index}")
            data.append(np.real(weight_history[:, selection_index]))
            columns.append(f"weight_im_{selection_index}")
            data.append(np.imag(weight_history[:, selection_index]))
        else:
            columns.append(f"weight_re_{selection_index}")
            data.append(np.real(weight_history[:, selection_index]))

    for array, prefix in _REAL_SELECTION_FIELDS:
        if prefix == "selected_gamma":
            history = selected_gamma_history
        else:
            history = {
                "selected_angle": selected_angle_history,
                "selected_smoothed_angle": selected_smoothed_angle_history,
                "selected_q": selected_q_history,
                "selected_chi": selected_chi_history,
                "selected_psi": selected_psi_history,
            }[prefix]
        if history is None:
            continue
        history = np.asarray(history)
        for selection_index in range(history.shape[1]):
            columns.append(f"{prefix}_{selection_index}")
            data.append(history[:, selection_index])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        rows = np.column_stack(data)
        for row in rows:
            writer.writerow([f"{value:.16e}" for value in row])


def _read_table(path: Path) -> tuple[list[str], np.ndarray]:
    """The CSV's header and its body as one ``(rows, columns)`` float table."""

    with path.open("rb") as file:
        header = file.readline().decode("utf-8").rstrip("\r\n").split(",")
        body = file.read()
    number_of_columns = len(header)
    if not body.strip():
        return header, np.zeros((0, number_of_columns))

    body = body.replace(b"\r\n", b"\n").rstrip(b"\n")
    number_of_rows = body.count(b"\n") + 1
    try:
        flat = np.fromstring(body.replace(b"\n", b","), sep=",")
    except ValueError:
        flat = None
    if flat is not None and flat.size == number_of_rows * number_of_columns:
        return header, flat.reshape(number_of_rows, number_of_columns)

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader]
    return header, np.asarray(rows, dtype=float)


def read_method_history(path: Path) -> dict[str, np.ndarray]:
    """Inverse of ``write_method_history``: rebuild each named array whose columns
    are present in the file."""

    header, table = _read_table(path)
    columns = {name: index for index, name in enumerate(header)}

    def column(name: str) -> np.ndarray:
        # Copy, not a view: a view keeps the whole (~20 MB) table alive.
        return table[:, columns[name]].copy()

    result: dict[str, np.ndarray] = {}
    result["train_loss"] = column("train_loss")
    result["test_loss"] = column("test_loss")
    if "train_accuracy" in columns:
        result["train_accuracy"] = column("train_accuracy")
    if "test_accuracy" in columns:
        result["test_accuracy"] = column("test_accuracy")

    diagnostic_indices = sorted(
        int(name.split("_")[1]) for name in header if name.startswith("diagnostic_")
    )
    if diagnostic_indices:
        result["diagnostics"] = np.column_stack(
            [column(f"diagnostic_{index}") for index in diagnostic_indices]
        )

    weight_real_indices = sorted(
        int(name.split("_")[-1]) for name in header if name.startswith("weight_re_")
    )
    if weight_real_indices:
        has_imag = any(name.startswith("weight_im_") for name in header)
        if has_imag:
            result["weight_history"] = np.column_stack(
                [
                    column(f"weight_re_{index}") + 1j * column(f"weight_im_{index}")
                    for index in weight_real_indices
                ]
            )
        else:
            result["weight_history"] = np.column_stack(
                [column(f"weight_re_{index}") for index in weight_real_indices]
            )

    for _, prefix in _REAL_SELECTION_FIELDS:
        field_name = {
            "selected_gamma": "selected_gamma_history",
            "selected_angle": "selected_angle_history",
            "selected_smoothed_angle": "selected_smoothed_angle_history",
            "selected_q": "selected_q_history",
            "selected_chi": "selected_chi_history",
            "selected_psi": "selected_psi_history",
        }[prefix]
        indices = sorted(
            int(name[len(prefix) + 1 :])
            for name in header
            if name.startswith(prefix + "_")
            # Exact "<prefix>_<int>" match: some field prefixes are prefixes of other fields.
            and name[len(prefix) + 1 :].isdigit()
        )
        if indices:
            result[field_name] = np.column_stack(
                [column(f"{prefix}_{index}") for index in indices]
            )

    return result
