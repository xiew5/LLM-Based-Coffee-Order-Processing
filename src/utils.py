from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER_NAME = "coffee_nlp"


@dataclass
class SchemaValidationResult:
    is_valid: bool
    errors: List[str]


def get_logger(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    outputs = base_dir / "outputs"
    paths = {
        "outputs": outputs,
        "predictions": outputs / "predictions",
        "metrics": outputs / "metrics",
        "error_analysis": outputs / "error_analysis",
        "logs": outputs / "logs",
        "raw_responses": outputs / "logs" / "raw_responses",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_string(text: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


def validate_item_schema(item: Any, idx: int = 0) -> List[str]:
    errors: List[str] = []
    if not isinstance(item, dict):
        return [f"items[{idx}] is not an object"]

    for key in ["name", "size", "quantity", "modifiers"]:
        if key not in item:
            errors.append(f"items[{idx}] missing '{key}'")

    if "name" in item and not isinstance(item["name"], str):
        errors.append(f"items[{idx}].name is not a string")

    if "size" in item and item["size"] is not None and not isinstance(item["size"], str):
        errors.append(f"items[{idx}].size must be string or null")

    if "quantity" in item:
        quantity = item["quantity"]
        if not isinstance(quantity, int) or quantity < 0:
            errors.append(f"items[{idx}].quantity must be non-negative integer")

    if "modifiers" in item:
        modifiers = item["modifiers"]
        if not isinstance(modifiers, list):
            errors.append(f"items[{idx}].modifiers is not a list")
        else:
            for m_idx, mod in enumerate(modifiers):
                if not isinstance(mod, str):
                    errors.append(f"items[{idx}].modifiers[{m_idx}] is not a string")

    return errors


def validate_order_schema(obj: Any) -> SchemaValidationResult:
    errors: List[str] = []
    if not isinstance(obj, dict):
        return SchemaValidationResult(False, ["root is not an object"])

    for key in ["items", "total_price"]:
        if key not in obj:
            errors.append(f"missing '{key}'")

    items = obj.get("items")
    if not isinstance(items, list):
        errors.append("'items' must be a list")
    else:
        for idx, item in enumerate(items):
            errors.extend(validate_item_schema(item, idx))

    total_price = safe_float(obj.get("total_price"))
    if total_price is None:
        errors.append("'total_price' must be numeric")

    return SchemaValidationResult(is_valid=(len(errors) == 0), errors=errors)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_modifier(modifier: str) -> str:
    return " ".join(modifier.strip().lower().split())


def canonical_item(item: Dict[str, Any]) -> Tuple[str, Optional[str], int, Tuple[str, ...]]:
    name = str(item.get("name", "")).strip().lower()
    size_raw = item.get("size")
    size = None if size_raw is None else str(size_raw).strip().lower()
    quantity = int(item.get("quantity", 0)) if str(item.get("quantity", "")).isdigit() else 0
    modifiers = tuple(sorted(normalize_modifier(m) for m in item.get("modifiers", []) if isinstance(m, str)))
    return (name, size, quantity, modifiers)


def canonical_order(order_obj: Dict[str, Any]) -> Dict[str, Any]:
    items = [canonical_item(i) for i in order_obj.get("items", []) if isinstance(i, dict)]
    items.sort()
    total_price = round(float(order_obj.get("total_price", 0.0)), 4)
    return {"items": items, "total_price": total_price}


def counter_from_items(items: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(canonical_item(item) for item in items if isinstance(item, dict))
