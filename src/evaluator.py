from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .utils import canonical_order, counter_from_items


def _total_price_equal(pred: float, gold: float, tol: float = 1e-2) -> bool:
    return abs(pred - gold) <= tol


def _pair_items_greedy(pred_items: List[Dict[str, Any]], gold_items: List[Dict[str, Any]]) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    remaining_gold = gold_items[:]
    pairs: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]] = []

    def score(a: Dict[str, Any], b: Dict[str, Any]) -> int:
        s = 0
        if str(a.get("name", "")).strip().lower() == str(b.get("name", "")).strip().lower():
            s += 2
        if (a.get("size") or None) == (b.get("size") or None):
            s += 1
        if int(a.get("quantity", 0)) == int(b.get("quantity", 0)):
            s += 1
        if set(map(str.lower, a.get("modifiers", []))) == set(map(str.lower, b.get("modifiers", []))):
            s += 1
        return s

    for pred_item in pred_items:
        if not remaining_gold:
            pairs.append((pred_item, None))
            continue

        best_idx = max(range(len(remaining_gold)), key=lambda i: score(pred_item, remaining_gold[i]))
        pairs.append((pred_item, remaining_gold.pop(best_idx)))

    for gold_item in remaining_gold:
        pairs.append((None, gold_item))

    return pairs


def evaluate_single(pred_obj: Optional[Dict[str, Any]], gold_obj: Dict[str, Any]) -> Dict[str, Any]:
    if pred_obj is None:
        return {
            "full_exact_match": 0,
            "item_match_precision": 0.0,
            "item_match_recall": 0.0,
            "item_match_f1": 0.0,
            "field_name_accuracy": 0.0,
            "field_size_accuracy": 0.0,
            "field_quantity_accuracy": 0.0,
            "field_modifiers_accuracy": 0.0,
            "total_price_exact": 0,
            "total_price_absolute_error": abs(float(gold_obj.get("total_price", 0.0))),
        }

    pred_canon = canonical_order(pred_obj)
    gold_canon = canonical_order(gold_obj)

    full_exact_match = int(pred_canon == gold_canon)

    pred_items = pred_obj.get("items", [])
    gold_items = gold_obj.get("items", [])

    pred_counter: Counter = counter_from_items(pred_items)
    gold_counter: Counter = counter_from_items(gold_items)
    overlap = sum((pred_counter & gold_counter).values())

    precision = overlap / max(1, sum(pred_counter.values()))
    recall = overlap / max(1, sum(gold_counter.values()))
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    pairs = _pair_items_greedy(pred_items, gold_items)
    denom = max(1, len(pairs))

    name_acc = 0.0
    size_acc = 0.0
    quantity_acc = 0.0
    modifiers_acc = 0.0

    for pred_item, gold_item in pairs:
        if pred_item is None or gold_item is None:
            continue

        if str(pred_item.get("name", "")).strip().lower() == str(gold_item.get("name", "")).strip().lower():
            name_acc += 1.0
        if (pred_item.get("size") or None) == (gold_item.get("size") or None):
            size_acc += 1.0
        if int(pred_item.get("quantity", 0)) == int(gold_item.get("quantity", 0)):
            quantity_acc += 1.0

        pmods = sorted(str(m).strip().lower() for m in pred_item.get("modifiers", []))
        gmods = sorted(str(m).strip().lower() for m in gold_item.get("modifiers", []))
        if pmods == gmods:
            modifiers_acc += 1.0

    pred_price = float(pred_obj.get("total_price", 0.0))
    gold_price = float(gold_obj.get("total_price", 0.0))

    return {
        "full_exact_match": full_exact_match,
        "item_match_precision": precision,
        "item_match_recall": recall,
        "item_match_f1": f1,
        "field_name_accuracy": name_acc / denom,
        "field_size_accuracy": size_acc / denom,
        "field_quantity_accuracy": quantity_acc / denom,
        "field_modifiers_accuracy": modifiers_acc / denom,
        "total_price_exact": int(_total_price_equal(pred_price, gold_price)),
        "total_price_absolute_error": abs(pred_price - gold_price),
    }


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    metric_keys = [
        "full_exact_match",
        "item_match_precision",
        "item_match_recall",
        "item_match_f1",
        "field_name_accuracy",
        "field_size_accuracy",
        "field_quantity_accuracy",
        "field_modifiers_accuracy",
        "total_price_exact",
        "total_price_absolute_error",
    ]

    return {
        key: sum(float(r.get(key, 0.0)) for r in rows) / len(rows)
        for key in metric_keys
    }


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored: List[Dict[str, Any]] = []

    for rec in records:
        metrics = evaluate_single(rec.get("pred_obj"), rec["gold_obj"])
        combined = {**rec, **metrics}
        scored.append(combined)

    overall = aggregate_metrics(scored)

    subset_names = ["correction", "cancellation", "negation", "multi_item", "long_context"]
    subset_metrics: Dict[str, Dict[str, float]] = {}
    for subset in subset_names:
        subset_rows = [r for r in scored if bool(r.get(subset, False))]
        subset_metrics[subset] = aggregate_metrics(subset_rows)

    return {
        "overall": overall,
        "subsets": subset_metrics,
        "scored_rows": scored,
    }
