from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import SchemaValidationResult, get_logger, parse_json_string, validate_order_schema


TEXT_COL_CANDIDATES = ["order", "text", "input", "utterance", "query"]
LABEL_COL_CANDIDATES = ["expected_json", "label", "target", "output_json", "gold_json"]
ID_COL_CANDIDATES = ["id", "row_id", "sample_id"]


@dataclass
class LoadedTrainData:
    df: pd.DataFrame
    text_col: str
    label_col: str
    id_col: str
    malformed_rows: List[Dict[str, Any]]


@dataclass
class LoadedTestData:
    df: pd.DataFrame
    text_col: str
    id_col: str


def _pick_column(columns: List[str], candidates: List[str], fallback_index: int) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return columns[fallback_index]


def _try_repair_json(raw_text: str) -> Tuple[Optional[Any], Optional[str], bool]:
    obj, err = parse_json_string(raw_text)
    if err is None:
        return obj, None, False

    repaired = raw_text.strip()
    if repaired.startswith("'") and repaired.endswith("'"):
        repaired = repaired[1:-1]

    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")

    obj2, err2 = parse_json_string(repaired)
    if err2 is None:
        return obj2, None, True

    # Last conservative attempt: python literal style single quotes.
    try:
        python_style = repaired
        if '"' not in python_style and "'" in python_style:
            python_style = python_style.replace("'", '"')
        obj3 = json.loads(python_style)
        return obj3, None, True
    except Exception as exc:  # noqa: BLE001
        return None, f"{err}; repair_failed={exc}", True


def load_train_data(path: Path) -> LoadedTrainData:
    logger = get_logger()
    df = pd.read_csv(path)

    text_col = _pick_column(list(df.columns), TEXT_COL_CANDIDATES, fallback_index=1)
    label_col = _pick_column(list(df.columns), LABEL_COL_CANDIDATES, fallback_index=len(df.columns) - 1)
    id_col = _pick_column(list(df.columns), ID_COL_CANDIDATES, fallback_index=0)

    malformed_rows: List[Dict[str, Any]] = []
    parsed_labels: List[Optional[Dict[str, Any]]] = []
    label_valid: List[bool] = []
    label_errors: List[str] = []
    repaired_flags: List[bool] = []

    for idx, row in df.iterrows():
        raw_label = str(row.get(label_col, "")).strip()
        obj, parse_err, repaired = _try_repair_json(raw_label)
        repaired_flags.append(repaired)

        if parse_err is not None or obj is None:
            parsed_labels.append(None)
            label_valid.append(False)
            label_errors.append(parse_err or "unable to parse label")
            malformed_rows.append(
                {
                    "row_index": int(idx),
                    "id": row.get(id_col),
                    "error": parse_err or "unable to parse label",
                    "raw_label": raw_label,
                }
            )
            continue

        validation: SchemaValidationResult = validate_order_schema(obj)
        parsed_labels.append(obj if validation.is_valid else None)
        label_valid.append(validation.is_valid)
        err_text = " | ".join(validation.errors)
        label_errors.append(err_text)

        if not validation.is_valid:
            malformed_rows.append(
                {
                    "row_index": int(idx),
                    "id": row.get(id_col),
                    "error": err_text,
                    "raw_label": raw_label,
                }
            )

    df = df.copy()
    df["parsed_label"] = parsed_labels
    df["label_is_valid"] = label_valid
    df["label_validation_errors"] = label_errors
    df["label_repaired"] = repaired_flags

    malformed_count = len(malformed_rows)
    if malformed_count:
        logger.warning("Found %d malformed labeled rows in train set.", malformed_count)
    else:
        logger.info("All training labels parsed and validated successfully.")

    return LoadedTrainData(
        df=df,
        text_col=text_col,
        label_col=label_col,
        id_col=id_col,
        malformed_rows=malformed_rows,
    )


def load_test_data(path: Path) -> LoadedTestData:
    df = pd.read_csv(path)
    text_col = _pick_column(list(df.columns), TEXT_COL_CANDIDATES, fallback_index=1)
    id_col = _pick_column(list(df.columns), ID_COL_CANDIDATES, fallback_index=0)
    return LoadedTestData(df=df, text_col=text_col, id_col=id_col)


def inspect_dataframe(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "text_col": text_col,
        "text_null_count": int(df[text_col].isna().sum()) if text_col in df.columns else None,
    }
    if text_col in df.columns:
        lengths = df[text_col].astype(str).map(lambda x: len(x.split()))
        summary["text_length_mean"] = float(lengths.mean())
        summary["text_length_p95"] = float(lengths.quantile(0.95))

    if label_col is not None and label_col in df.columns:
        summary["label_col"] = label_col
        summary["label_null_count"] = int(df[label_col].isna().sum())

    return summary
