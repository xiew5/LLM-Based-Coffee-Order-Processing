from __future__ import annotations

import re
import random
from typing import Dict, List, Tuple

import pandas as pd


CORRECTION_RE = re.compile(
    r"\b(actually|wait|i mean|make that|change that to|bump that to)\b", re.IGNORECASE
)
CANCELLATION_RE = re.compile(
    r"\b(cancel|scratch that|nevermind|never mind|remove that|drop that)\b", re.IGNORECASE
)
NEGATION_RE = re.compile(
    r"\b(no\b|without|don't|do not|hold the|no ice|no whip|not)\b", re.IGNORECASE
)
MULTI_ITEM_RE = re.compile(
    r"\b(and|plus|also|paired with|coupled with|next|then|oh, and)\b", re.IGNORECASE
)


def tag_phenomena(text: str, long_context_threshold: int = 30) -> Dict[str, bool]:
    normalized = str(text)
    tokens = normalized.split()
    return {
        "correction": bool(CORRECTION_RE.search(normalized)),
        "cancellation": bool(CANCELLATION_RE.search(normalized)),
        "negation": bool(NEGATION_RE.search(normalized)),
        "multi_item": bool(MULTI_ITEM_RE.search(normalized)),
        "long_context": len(tokens) >= long_context_threshold,
    }


def add_subset_tags(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = df.copy()
    tags = out[text_col].astype(str).map(tag_phenomena)
    out["subset_tags"] = tags
    for col in ["correction", "cancellation", "negation", "multi_item", "long_context"]:
        out[col] = tags.map(lambda t: bool(t.get(col, False)))
    out["tag_signature"] = out[["correction", "cancellation", "negation", "multi_item", "long_context"]].astype(int).astype(str).agg("".join, axis=1)
    return out


def _stratified_indices(
    signatures: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[int]] = {}
    for idx, sig in enumerate(signatures):
        grouped.setdefault(sig, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for _, idxs in grouped.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)

        if len(shuffled) == 1:
            train_idx.extend(shuffled)
            continue

        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, len(shuffled) - 1)
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    if stratify and "tag_signature" in df.columns:
        train_idx, val_idx = _stratified_indices(df["tag_signature"].astype(str).tolist(), val_ratio, seed)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
    else:
        rng = random.Random(seed)
        indices = list(range(len(df)))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(df) * val_ratio)))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


def kfold_splits(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
    stratify: bool = True,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    if df.empty:
        return []
    if k < 2:
        raise ValueError("k must be >= 2 for k-fold cross-validation")
    if len(df) < k:
        raise ValueError(f"Dataset size ({len(df)}) must be >= k ({k})")

    rng = random.Random(seed)
    folds: List[List[int]] = [[] for _ in range(k)]

    if stratify and "tag_signature" in df.columns:
        grouped: Dict[str, List[int]] = {}
        for idx, sig in enumerate(df["tag_signature"].astype(str).tolist()):
            grouped.setdefault(sig, []).append(idx)

        for idxs in grouped.values():
            shuffled = idxs[:]
            rng.shuffle(shuffled)
            for i, idx in enumerate(shuffled):
                folds[i % k].append(idx)
    else:
        indices = list(range(len(df)))
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    all_idx = set(range(len(df)))
    for fold_id in range(k):
        val_idx = sorted(folds[fold_id])
        train_idx = sorted(all_idx.difference(val_idx))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        splits.append((train_df, val_df))

    return splits
