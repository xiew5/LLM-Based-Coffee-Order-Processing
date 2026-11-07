from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SCHEMA_DESCRIPTION = {
    "items": [
        {
            "name": "string",
            "size": "string or null",
            "quantity": "integer >= 0",
            "modifiers": ["string", "..."],
        }
    ],
    "total_price": "number",
}


def load_menu_text(menu_path: Path) -> str:
    return menu_path.read_text(encoding="utf-8")


def sample_few_shots(
    train_df: pd.DataFrame,
    text_col: str,
    gold_obj_col: str,
    n_examples: int = 2,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    valid_df = train_df[train_df[gold_obj_col].notna()].copy()
    if valid_df.empty or n_examples <= 0:
        return []

    rng = random.Random(seed)
    sample_n = min(n_examples, len(valid_df))
    idxs = rng.sample(list(valid_df.index), sample_n)
    rows = valid_df.loc[idxs]

    examples: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        examples.append(
            {
                "order": str(row[text_col]),
                "output": row[gold_obj_col],
            }
        )
    return examples


def build_prompt(
    order_text: str,
    menu_text: str,
    few_shots: Optional[List[Dict[str, Any]]] = None,
    include_few_shot: bool = True,
) -> Dict[str, str]:
    few_shots = few_shots or []
    examples_block = ""

    if include_few_shot and few_shots:
        rendered = []
        for ex in few_shots:
            rendered.append(
                "Example Input:\n"
                f"{ex['order']}\n"
                "Example Output (JSON only):\n"
                f"{json.dumps(ex['output'], ensure_ascii=False)}"
            )
        examples_block = "\n\n".join(rendered)

    system_prompt = (
        "You are an information extraction model for coffee orders. "
        "Return valid JSON only, no markdown, no code fences, no explanations."
    )

    user_parts = [
        "Menu and pricing rules:",
        menu_text,
        "",
        "Output schema (must be exact):",
        json.dumps(SCHEMA_DESCRIPTION, ensure_ascii=False),
        "",
        "Instructions:",
        "1) Output a JSON object with keys: items, total_price.",
        "2) Each item must include: name, size, quantity, modifiers.",
        "3) Use null for size on food items.",
        "4) Respect corrections/cancellations/negations in the order text.",
        "5) Compute total_price numerically from menu rules.",
        "6) Return strictly valid JSON only.",
    ]

    if examples_block:
        user_parts += ["", "Few-shot examples:", examples_block]

    user_parts += ["", "Now parse this order:", order_text]
    user_prompt = "\n".join(user_parts)

    return {"system_prompt": system_prompt, "user_prompt": user_prompt}
