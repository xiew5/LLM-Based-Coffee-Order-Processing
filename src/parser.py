from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .utils import SchemaValidationResult, validate_order_schema


@dataclass
class ParseResult:
    parsed_obj: Optional[Dict[str, Any]]
    is_valid: bool
    repair_used: bool
    errors: List[str]
    extracted_text: str


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _extract_candidate_json(text: str) -> str:
    stripped = text.strip()

    fence_match = JSON_FENCE_RE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Fallback: first '{' to last '}'
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    return stripped


def _minor_json_repairs(candidate: str) -> str:
    repaired = candidate.strip()
    # Remove trailing commas before ] or }
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    # Normalize smart quotes
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    return repaired


def parse_and_validate_model_output(output_text: str) -> ParseResult:
    errors: List[str] = []
    extracted = _extract_candidate_json(output_text)

    try:
        obj = json.loads(extracted)
        repaired = False
    except json.JSONDecodeError as exc:
        errors.append(f"json_decode_error: {exc}")
        repaired_text = _minor_json_repairs(extracted)
        repaired = repaired_text != extracted

        try:
            obj = json.loads(repaired_text)
            extracted = repaired_text
        except json.JSONDecodeError as exc2:
            errors.append(f"json_decode_error_after_repair: {exc2}")
            return ParseResult(
                parsed_obj=None,
                is_valid=False,
                repair_used=repaired,
                errors=errors,
                extracted_text=extracted,
            )

    validation: SchemaValidationResult = validate_order_schema(obj)
    if not validation.is_valid:
        errors.extend([f"schema_error: {e}" for e in validation.errors])

    return ParseResult(
        parsed_obj=obj if validation.is_valid else None,
        is_valid=validation.is_valid,
        repair_used=repaired,
        errors=errors,
        extracted_text=extracted,
    )
