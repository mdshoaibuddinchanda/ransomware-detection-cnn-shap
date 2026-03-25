"""Clean-checkout validation for the preserved legacy project."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = (
    "instructions", "LLC-stores", "L1-icache-load-misses", "branch-load-misses",
    "node-load-misses", "rd_req", "rd_bytes", "wr_req", "wr_bytes",
    "flush_operations", "rd_total_times", "wr_total_times", "flush_total_times",
)


def validate_csv(path: Path, *, require_label: bool) -> dict[str, object]:
    """Validate one checked-in CSV and return a reproducible summary."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        expected = FEATURE_COLUMNS + (("label",) if require_label else ())
        if columns != expected:
            raise AssertionError(f"{path}: expected columns {expected}, found {columns}")
        rows = 0
        labels: Counter[str] = Counter()
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            for column in FEATURE_COLUMNS:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise AssertionError(f"{path}:{row_number}: non-numeric {column}") from exc
                if not math.isfinite(value):
                    raise AssertionError(f"{path}:{row_number}: non-finite {column}")
            if require_label:
                label = row["label"]
                if label not in {"0", "1"}:
                    raise AssertionError(f"{path}:{row_number}: invalid label {label!r}")
                labels[label] += 1
    if rows == 0:
        raise AssertionError(f"{path}: empty dataset")
    return {"path": path.relative_to(ROOT).as_posix(), "rows": rows, "labels": dict(labels)}


def validate_project() -> dict[str, object]:
    """Validate source, notebook, required inputs, and data contracts."""
    required = [
        ROOT / "Main.py", ROOT / "run.bat", ROOT / "Requirements.txt",
        ROOT / "Dataset" / "hpc_io_data.csv", ROOT / "Dataset" / "testData.csv",
        ROOT / "Ransomware_Paper_Enhancements.ipynb",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing required project files: {missing}")
    source = (ROOT / "Main.py").read_text(encoding="utf-8")
    ast.parse(source, filename="Main.py")
    compile(source, "Main.py", "exec")
    with (ROOT / "Ransomware_Paper_Enhancements.ipynb").open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    if notebook.get("nbformat") not in {4, 5} or not isinstance(notebook.get("cells"), list):
        raise AssertionError("notebook is not a valid nbformat 4/5 document")

    required_source_contract = (
        "RANDOM_STATE = 42",
        "X_train_raw, X_val_raw, y_train, y_val = train_test_split",
        "scaler.fit_transform(X_train_raw)",
        "validation_data=(x_val_dl, y_val_dl)",
        "validation_data=(x_val_lstm, y_val_dl)",
        "validation_data=(x_val_cnn, y_val_dl)",
        "_record_model_metadata",
    )
    missing_contract = [token for token in required_source_contract if token not in source]
    if missing_contract:
        raise AssertionError(f"Main.py is missing audit contract: {missing_contract}")
    prohibited_source_patterns = (
        r"predict\s*\[\s*0\s*:\s*9500\s*\]\s*=",
        r"scaler\.fit_transform\(X\s*\)",
        r"validation_data\s*=\s*\(\s*x_test",
    )
    for pattern in prohibited_source_patterns:
        if re.search(pattern, source, flags=re.IGNORECASE):
            raise AssertionError(f"Main.py contains prohibited audit pattern: {pattern}")

    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for token in ("Fallback model accuracy", "Using RandomForest fallback", "max(acc -"):
        if token in notebook_source:
            raise AssertionError(f"notebook contains unsupported evidence pattern: {token}")
    if "run_adversarial_robustness" in notebook_source:
        raise AssertionError("notebook still labels controlled noise as adversarial robustness")
    if any(cell.get("outputs") for cell in notebook["cells"] if cell.get("cell_type") == "code"):
        raise AssertionError("notebook contains stale outputs; execute it to create fresh evidence")
    return {
        "source": "Main.py",
        "notebook_cells": len(notebook["cells"]),
        "audit_contract": "passed",
        "datasets": [
            validate_csv(ROOT / "Dataset" / "hpc_io_data.csv", require_label=True),
            validate_csv(ROOT / "Dataset" / "testData.csv", require_label=False),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(validate_project(), indent=2, sort_keys=True))
