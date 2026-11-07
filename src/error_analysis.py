from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List


def categorize_error(record: Dict[str, Any]) -> List[str]:
    categories: List[str] = []

    if not bool(record.get("parse_valid", True)):
        categories.append("invalid_json")
        return categories

    pred_obj = record.get("pred_obj") or {"items": [], "total_price": 0.0}
    gold_obj = record.get("gold_obj") or {"items": [], "total_price": 0.0}

    pred_items = pred_obj.get("items", [])
    gold_items = gold_obj.get("items", [])

    if len(pred_items) < len(gold_items):
        categories.append("missing_item")
    if len(pred_items) > len(gold_items):
        categories.append("extra_item")

    n = min(len(pred_items), len(gold_items))
    for i in range(n):
        p = pred_items[i]
        g = gold_items[i]
        if int(p.get("quantity", 0)) != int(g.get("quantity", 0)):
            categories.append("wrong_quantity")
        if (p.get("size") or None) != (g.get("size") or None):
            categories.append("wrong_size")

        pmods = sorted(str(x).strip().lower() for x in p.get("modifiers", []))
        gmods = sorted(str(x).strip().lower() for x in g.get("modifiers", []))
        if pmods != gmods:
            categories.append("wrong_modifier")

    pred_price = float(pred_obj.get("total_price", 0.0))
    gold_price = float(gold_obj.get("total_price", 0.0))
    if abs(pred_price - gold_price) > 1e-2:
        categories.append("wrong_total_price")

    if record.get("negation", False) and record.get("full_exact_match", 0) == 0:
        categories.append("negation_failure")
    if record.get("cancellation", False) and record.get("full_exact_match", 0) == 0:
        categories.append("cancellation_failure")
    if record.get("correction", False) and record.get("full_exact_match", 0) == 0:
        categories.append("correction_failure")

    return sorted(set(categories))


def analyze_errors(
    scored_rows: List[Dict[str, Any]],
    max_examples_per_type: int = 10,
) -> Dict[str, Any]:
    category_counts: DefaultDict[str, int] = defaultdict(int)
    examples: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in scored_rows:
        categories = categorize_error(row)
        row["error_categories"] = categories
        for cat in categories:
            category_counts[cat] += 1
            if len(examples[cat]) < max_examples_per_type:
                examples[cat].append(
                    {
                        "id": row.get("id"),
                        "text": row.get("text"),
                        "gold_obj": row.get("gold_obj"),
                        "pred_obj": row.get("pred_obj"),
                        "parse_errors": row.get("parse_errors"),
                        "strategy": row.get("strategy"),
                    }
                )

    return {
        "counts": dict(sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))),
        "examples": dict(examples),
        "rows": scored_rows,
    }
