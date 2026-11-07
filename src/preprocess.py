from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional

NUMBER_WORD_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "single": "1",
    "double": "2",
    "triple": "3",
    "couple": "2",
    "pair": "2",
    "few": "3",
}

SIZES = ["short", "tall", "grande", "venti", "trenta", "small", "medium", "large"]
TEMPS = ["hot", "iced", "extra hot"]
MILKS = ["oat milk", "almond milk", "soy milk", "coconut milk", "breve", "skim milk", "whole milk"]
FLAVORS = ["vanilla", "caramel", "hazelnut", "peppermint", "sugar free vanilla", "classic syrup"]
MODIFIERS = [
    "whipped cream",
    "whip cream",
    "whip",
    "ice",
    "foam",
    "cold foam",
    "extra shot",
    "caramel drizzle",
] + FLAVORS + MILKS

ITEMS = [
    "caramel macchiato",
    "flat white",
    "chai latte",
    "matcha latte",
    "hot chocolate",
    "drip coffee",
    "cold brew",
    "iced coffee",
    "earl grey tea",
    "green tea",
    "strawberry smoothie",
    "frappe (mocha)",
    "frappe (coffee)",
    "frappe",
    "cappuccino",
    "americano",
    "espresso",
    "latte",
    "mocha",
    "butter croissant",
    "blueberry muffin",
    "bagel",
    "avocado toast",
    "bacon gouda sandwich",
]

CORRECTION_TRIGGERS = [
    "actually",
    "sorry",
    "wait",
    "i mean",
    "make that",
    "instead",
    "rather",
    "on second thought",
    "change that to",
]


def _strip_outer_whitespace(text: str) -> str:
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_outer_whitespace(text))


def _find_first_phrase(text: str, phrases: List[str]) -> Optional[str]:
    lower = text.lower()
    for phrase in sorted(phrases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            return phrase
    return None


def _extract_quantity(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", text)
    return int(m.group(1)) if m else None


def _extract_many(text: str, vocab: List[str]) -> List[str]:
    lower = text.lower()
    return [tok for tok in sorted(vocab, key=len, reverse=True) if re.search(rf"\b{re.escape(tok)}\b", lower)]


def _canonical_modifier(text: str) -> str:
    t = text.strip().lower()
    if t in {"whip", "whip cream"}:
        return "whipped cream"
    return t


def raw(text: str) -> str:
    return _strip_outer_whitespace(text)


def normalize_numbers(text: str) -> str:
    out = _normalize_whitespace(text)
    out = re.sub(r"\bcouple\s+of\b", "2", out, flags=re.IGNORECASE)
    out = re.sub(r"\bpair\s+of\b", "2", out, flags=re.IGNORECASE)

    out = re.sub(
        r"\b(" + "|".join(re.escape(k) for k in NUMBER_WORD_MAP.keys()) + r")\b",
        lambda m: NUMBER_WORD_MAP.get(m.group(1).lower(), m.group(1)),
        out,
        flags=re.IGNORECASE,
    )

    quant_heads = SIZES + ["latte", "americano", "espresso", "mocha", "cappuccino", "frappe", "tea", "coffee", "muffin", "bagel", "sandwich", "toast", "croissant"]
    out = re.sub(
        r"\b(a|an)\s+(?=(" + "|".join(re.escape(x) for x in quant_heads) + r")\b)",
        "1 ",
        out,
        flags=re.IGNORECASE,
    )
    return _normalize_whitespace(out)


def resolve_corrections(text: str) -> str:
    out = normalize_numbers(text)
    trigger_group = "|".join(re.escape(t) for t in CORRECTION_TRIGGERS)
    slot_values = sorted(set(SIZES + TEMPS + MILKS + FLAVORS), key=len, reverse=True)
    slot_group = "|".join(re.escape(v) for v in slot_values)

    out = re.sub(
        rf"\b({slot_group})\b\s*(?:,\s*)?(?:{trigger_group})\s+\b({slot_group})\b",
        lambda m: m.group(2),
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        rf"\b(\d+)\s+([a-z][a-z\s()]+?)\s+(?:no|actually|wait|sorry|make that)\s+(\d+)\s+\2\b",
        lambda m: f"{m.group(3)} {m.group(2)}",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(rf"\b({trigger_group})\b", " ", out, flags=re.IGNORECASE)
    return _normalize_whitespace(out)


def resolve_negation_cancellation(text: str) -> str:
    out = _normalize_whitespace(text)
    out = re.sub(r"\bhold the\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", r"without \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\bno\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", r"without \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\bremove\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", r"without \1", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(do not|don't)\s+add\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", r"without \2", out, flags=re.IGNORECASE)
    out = re.sub(r"\bcancel\s+(the\s+)?([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", r"cancel \2", out, flags=re.IGNORECASE)
    return _normalize_whitespace(out)


def _render_item(state: Dict[str, object]) -> str:
    qty = int(state.get("quantity", 1))
    size = state.get("size")
    temp = state.get("temperature")
    name = str(state.get("name", "")).strip()
    mods = [str(x) for x in state.get("mods", [])]
    rems = [str(x) for x in state.get("removals", [])]

    parts = [str(qty)]
    if size:
        parts.append(str(size))
    if temp:
        parts.append(str(temp))
    parts.append(name)

    out = " ".join(p for p in parts if p)
    if mods:
        out += " with " + ", ".join(mods)
    if rems:
        out += " no " + ", ".join(rems)
    return out.strip()


def stateful_order_rewrite(text: str) -> str:
    source = resolve_negation_cancellation(resolve_corrections(normalize_numbers(text)))
    clauses = [c.strip() for c in re.split(r"[.;]|\bthen\b|\bnext\b", source, flags=re.IGNORECASE) if c.strip()]

    states: List[Dict[str, object]] = []
    cur_idx: Optional[int] = None

    for clause in clauses:
        clause_l = clause.lower()

        cancel_item = re.search(r"\bcancel\s+([a-z][a-z\s()]+)$", clause_l)
        if cancel_item:
            target = cancel_item.group(1).strip()
            states = [s for s in states if target not in str(s.get("name", ""))]
            cur_idx = (len(states) - 1) if states else None
            continue

        item = _find_first_phrase(clause, ITEMS)
        if item:
            states.append(
                {
                    "name": item,
                    "quantity": _extract_quantity(clause) or 1,
                    "size": _find_first_phrase(clause, SIZES),
                    "temperature": _find_first_phrase(clause, TEMPS),
                    "mods": [],
                    "removals": [],
                }
            )
            cur_idx = len(states) - 1

        if cur_idx is None:
            continue

        cur = states[cur_idx]
        qty = _extract_quantity(clause)
        if qty is not None:
            cur["quantity"] = qty

        size = _find_first_phrase(clause, SIZES)
        if size:
            cur["size"] = size

        temp = _find_first_phrase(clause, TEMPS)
        if temp:
            cur["temperature"] = temp

        for mod in _extract_many(clause, MODIFIERS):
            can = _canonical_modifier(mod)
            mods = list(cur.get("mods", []))
            if can not in mods:
                mods.append(can)
            cur["mods"] = mods

        for neg in re.findall(r"\bwithout\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", clause_l):
            rem = _canonical_modifier(neg)
            rems = list(cur.get("removals", []))
            if rem not in rems:
                rems.append(rem)
            cur["removals"] = rems
            cur["mods"] = [m for m in list(cur.get("mods", [])) if m != rem]

    if not states:
        return _normalize_whitespace(source)
    return "; ".join(_render_item(s) for s in states if str(s.get("name", "")).strip())


def edit_operation_linearized(text: str) -> str:
    source = resolve_negation_cancellation(resolve_corrections(normalize_numbers(text)))
    clauses = [c.strip() for c in re.split(r"[.;]|\bthen\b|\bnext\b", source, flags=re.IGNORECASE) if c.strip()]
    ops: List[str] = []

    for clause in clauses:
        clause_l = clause.lower()
        item = _find_first_phrase(clause, ITEMS)
        if item:
            ops.append(f"ADD_ITEM {item}")

        qty = _extract_quantity(clause)
        if qty is not None:
            ops.append(f"SET_QUANTITY {qty}")

        size = _find_first_phrase(clause, SIZES)
        if size:
            ops.append(f"SET_SIZE {size}")

        temp = _find_first_phrase(clause, TEMPS)
        if temp:
            ops.append(f"SET_TEMPERATURE {temp}")

        for mod in _extract_many(clause, MODIFIERS):
            ops.append(f"ADD_MODIFIER {_canonical_modifier(mod)}")

        for neg in re.findall(r"\bwithout\s+([a-z][a-z\s]+?)(?=[,.;]|\band\b|$)", clause_l):
            ops.append(f"REMOVE_MODIFIER {_canonical_modifier(neg)}")

        cancel = re.search(r"\bcancel\s+([a-z][a-z\s()]+)$", clause_l)
        if cancel:
            ops.append(f"CANCEL_ITEM {cancel.group(1).strip()}")

        if any(t in clause_l for t in CORRECTION_TRIGGERS):
            slot = _find_first_phrase(clause, SIZES + TEMPS + MILKS + FLAVORS)
            if slot:
                ops.append(f"REPLACE_VALUE {slot}")

    return "\n".join(ops) if ops else "NO_OP"


def full_stateful_normalized(text: str) -> str:
    out = normalize_numbers(text)
    out = resolve_corrections(out)
    out = resolve_negation_cancellation(out)
    out = stateful_order_rewrite(out)
    return _normalize_whitespace(out)


STRATEGY_REGISTRY: Dict[str, Callable[[str], str]] = {
    "raw": raw,
    "normalize_numbers": normalize_numbers,
    "resolve_corrections": resolve_corrections,
    "resolve_negation_cancellation": resolve_negation_cancellation,
    "stateful_order_rewrite": stateful_order_rewrite,
    "edit_operation_linearized": edit_operation_linearized,
    "full_stateful_normalized": full_stateful_normalized,
}


def apply_strategy(text: str, strategy_name: str) -> str:
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    return STRATEGY_REGISTRY[strategy_name](text)


def apply_strategies_batch(texts: Iterable[str], strategy_names: List[str]) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {name: [] for name in strategy_names}
    for text in texts:
        for name in strategy_names:
            results[name].append(apply_strategy(text, name))
    return results
