"""
Multi-agent structured extraction from images with Google Gemini.

extract_parallel() sends the same images, prompt and JSON Schema to N agents in parallel,
then merges their answers field by field:
  - <field>    the value most agents agree on
  - <field>_%  the share of agents that gave that value (1.0 = all agree)
  - <field>_opts  the different values proposed, with their share (only when there are several agents)
Long enums (more than 50 values) are removed from the schema sent to the model and remapped afterwards
with fuzzy matching, so the model is not slowed down by huge enum lists.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator
from PIL import Image, ImageOps
from rapidfuzz import fuzz, process
from shared.client_resilience import RetryPolicy, is_transient_google_error, run_with_retries


Provider = Literal["google"]
JSONDict = dict[str, Any]


class ExtractionError(RuntimeError):
    pass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSettings:
    provider: Provider
    model: str

    # Prompts
    system_prompt: str
    user_prompt: str

    # Sampling (None = provider default)
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    thinking_level: Literal["low", "medium", "high"] | None = None


MAX_IMAGES = 20


MAX_IMAGE_BYTES = 1_000_000


LONG_ENUM_THRESHOLD = 50


FUZZY_REFINE_RE = re.compile(r"^fuzzy_refine(?:_(\d+))?$", re.IGNORECASE)


AI_PROVIDER_RETRY_POLICY = RetryPolicy(
    max_attempts=max(1, int(os.getenv("AI_CLIENT_RETRY_ATTEMPTS", "2"))),
    base_delay_seconds=max(0.0, float(os.getenv("AI_CLIENT_RETRY_BASE_DELAY_SEC", "0.5"))),
    max_delay_seconds=max(0.0, float(os.getenv("AI_CLIENT_RETRY_MAX_DELAY_SEC", "4.0"))),
    jitter_ratio=0.15,
)


@dataclass(frozen=True)
class EnumRule:
    # Path segments to the target field in the *output JSON*.
    # Uses "*" as wildcard for arrays (apply to each item).
    path: tuple[str | Literal["*"], ...]

    # The original enum list removed from schema.
    enum_list: list[Any]


@dataclass(frozen=True)
class FuzzyEnumRule:
    # Path segments to the target field in the *output JSON*.
    # Uses "*" as wildcard for arrays (apply to each item).
    path: tuple[str | Literal["*"], ...]

    # The original enum list removed from schema.
    enum_list: list[Any]

    # Number of nearest enum candidates to keep per agent output.
    top_k: int = 1


def _parse_fuzzy_refine_top_k(raw: Any) -> int | None:
    if not isinstance(raw, str):
        return None
    m = FUZZY_REFINE_RE.fullmatch(raw.strip())
    if not m:
        return None
    raw_k = m.group(1)
    if not raw_k:
        return 1
    try:
        return max(1, int(raw_k))
    except Exception:
        return 1


def _json_pointer_get(root: Any, pointer: str) -> Any:
    # Supports only local refs like "#/properties/..." (JSON Pointer)
    if not pointer.startswith("#"):
        return None
    p = pointer[1:]
    if p.startswith("/"):
        p = p[1:]
    if not p:
        return root
    node = root
    for part in p.split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _schema_walk(
    schema_root: dict[str, Any],
    node: Any,
    path: list[str | Literal["*"]],
    *,
    visited_ref: set[str],
    visit_node: Callable[[dict[str, Any], tuple[str | Literal["*"], ...]], None],
) -> None:
    if not isinstance(node, dict):
        return

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#"):
        if ref in visited_ref:
            return
        visited_ref.add(ref)
        target = _json_pointer_get(schema_root, ref)
        if target is not None:
            _schema_walk(
                schema_root,
                target,
                path,
                visited_ref=visited_ref,
                visit_node=visit_node,
            )
        return

    visit_node(node, tuple(path))

    props = node.get("properties")
    if isinstance(props, dict):
        for k, v in props.items():
            _schema_walk(
                schema_root,
                v,
                path + [k],
                visited_ref=visited_ref,
                visit_node=visit_node,
            )

    items = node.get("items")
    if isinstance(items, dict):
        _schema_walk(
            schema_root,
            items,
            path + ["*"],
            visited_ref=visited_ref,
            visit_node=visit_node,
        )
    elif isinstance(items, list):
        for it in items:
            _schema_walk(
                schema_root,
                it,
                path + ["*"],
                visited_ref=visited_ref,
                visit_node=visit_node,
            )

    for comb in ("allOf", "anyOf", "oneOf"):
        lst = node.get(comb)
        if isinstance(lst, list):
            for it in lst:
                _schema_walk(
                    schema_root,
                    it,
                    path,
                    visited_ref=visited_ref,
                    visit_node=visit_node,
                )

    for defs_key in ("$defs", "definitions"):
        defs = node.get(defs_key)
        if isinstance(defs, dict):
            for _, v in defs.items():
                _schema_walk(
                    schema_root,
                    v,
                    path,
                    visited_ref=visited_ref,
                    visit_node=visit_node,
                )


def _schema_walk_collect_enum_rules(
    schema_root: dict[str, Any],
    node: Any,
    path: list[str | Literal["*"]],
    *,
    out_long_rules: list[EnumRule],
    out_fuzzy_rules: list[FuzzyEnumRule],
    visited_ref: set[str],
) -> None:
    def visit_node(current_node: dict[str, Any], current_path: tuple[str | Literal["*"], ...]) -> None:
        enum_list = current_node.get("enum")
        fuzzy_top_k = _parse_fuzzy_refine_top_k(current_node.get("enum_postprocess"))
        if not isinstance(enum_list, list):
            return
        if fuzzy_top_k is not None:
            out_fuzzy_rules.append(
                FuzzyEnumRule(
                    path=current_path,
                    enum_list=list(enum_list),
                    top_k=min(max(1, fuzzy_top_k), max(1, len(enum_list))),
                )
            )
            current_node.pop("enum", None)
            current_node.pop("enum_postprocess", None)
            return
        if len(enum_list) > LONG_ENUM_THRESHOLD:
            out_long_rules.append(EnumRule(path=current_path, enum_list=list(enum_list)))
            current_node.pop("enum", None)

    _schema_walk(
        schema_root,
        node,
        path,
        visited_ref=visited_ref,
        visit_node=visit_node,
    )


def _schema_walk_collect_enum_paths(
    schema_root: dict[str, Any],
    node: Any,
    path: list[str | Literal["*"]],
    *,
    out_paths: list[tuple[str | Literal["*"], ...]],
    visited_ref: set[str],
) -> None:
    def visit_node(current_node: dict[str, Any], current_path: tuple[str | Literal["*"], ...]) -> None:
        enum_list = current_node.get("enum")
        if isinstance(enum_list, list):
            out_paths.append(current_path)

    _schema_walk(
        schema_root,
        node,
        path,
        visited_ref=visited_ref,
        visit_node=visit_node,
    )


def relax_schema_enums(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[EnumRule], list[FuzzyEnumRule]]:
    """
    Returns:
      - relaxed_schema: deep copy of schema with long enums removed
      - enum_rules: long enums removed so we can remap post-hoc
      - fuzzy_enum_rules: enum_postprocess fuzzy rules
    """
    # Deep copy without importing deepcopy (keeps dependencies minimal).
    relaxed = json.loads(json.dumps(schema))
    long_rules: list[EnumRule] = []
    fuzzy_rules: list[FuzzyEnumRule] = []
    _schema_walk_collect_enum_rules(
        relaxed,
        relaxed,
        [],
        out_long_rules=long_rules,
        out_fuzzy_rules=fuzzy_rules,
        visited_ref=set(),
    )
    return relaxed, long_rules, fuzzy_rules


def collect_enforced_enum_paths(
    schema: dict[str, Any],
) -> list[tuple[str | Literal["*"], ...]]:
    """
    Collects enum-constrained field paths from the schema currently sent to the model.
    If relax_schema_enums() already removed a field enum, that path is not returned here.
    """
    raw_paths: list[tuple[str | Literal["*"], ...]] = []
    _schema_walk_collect_enum_paths(
        schema,
        schema,
        [],
        out_paths=raw_paths,
        visited_ref=set(),
    )
    seen: set[tuple[str | Literal["*"], ...]] = set()
    out: list[tuple[str | Literal["*"], ...]] = []
    for p in raw_paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def validate_json_schema(schema: dict[str, Any]) -> None:
    # Structured output needs a JSON Schema whose root is an object.
    if schema.get("type") != "object":
        raise ExtractionError(
            "Invalid extraction schema: root must be JSON Schema with type 'object'. "
            "If you passed a function/tool wrapper, pass wrapper['parameters'] (or let this module unwrap it)."
        )
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        raise ExtractionError(f"Invalid JSON schema (Draft 2020-12): {e}") from e


def normalize_extraction_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Accepts either:
      - raw JSON Schema (root type=object), or
      - function/tool-style wrapper: {"name", "description", "parameters": {...schema...}}
    Returns the raw JSON Schema object.
    """
    if not isinstance(schema, dict):
        raise ExtractionError("Schema must be a dict.")

    params = schema.get("parameters")
    if isinstance(params, dict):
        # If top-level does not look like a JSON Schema, treat it as wrapper.
        schema_keys = {"type", "properties", "items", "allOf", "anyOf", "oneOf", "$ref", "$defs", "definitions"}
        if not any(k in schema for k in schema_keys):
            return params

    return schema


def _detect_image_mime(image_bytes: bytes) -> str:
    try:
        im = Image.open(io.BytesIO(image_bytes))
        fmt = (im.format or "").upper()
        if fmt == "PNG":
            return "image/png"
        if fmt in ("JPG", "JPEG"):
            return "image/jpeg"
        if fmt == "WEBP":
            return "image/webp"
        if fmt == "TIFF":
            return "image/tiff"
    except Exception:
        pass
    # Default
    return "image/jpeg"


def compress_image_under_limit(
    image_bytes: bytes,
    *,
    target_bytes: int = MAX_IMAGE_BYTES,
    min_quality: int = 40,
    start_quality: int = 85,
    max_dim: int = 2200,
) -> bytes:
    """
    Converts to JPEG (RGB) and compresses/downscales until <= target_bytes.
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        im = ImageOps.exif_transpose(im)
    except Exception as e:
        raise ExtractionError(f"Invalid image bytes: {e}") from e

    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    elif im.mode == "L":
        im = im.convert("RGB")

    # Optional downscale to cap dimensions early.
    w, h = im.size
    max_side = max(w, h)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    # If already under target and seems jpeg-ish, keep original to avoid extra CPU.
    if len(image_bytes) <= target_bytes and _detect_image_mime(image_bytes) == "image/jpeg":
        return image_bytes

    # Try decreasing quality.
    quality = start_quality
    for _ in range(8):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True, progressive=True)
        out = buf.getvalue()
        if len(out) <= target_bytes:
            return out
        quality = max(min_quality, int(quality * 0.85))

    # If still too big, downscale iteratively.
    for _ in range(6):
        w, h = im.size
        if w <= 600 or h <= 600:
            break
        im = im.resize((max(1, int(w * 0.85)), max(1, int(h * 0.85))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(min_quality), optimize=True, progressive=True)
        out = buf.getvalue()
        if len(out) <= target_bytes:
            return out

    # Worst-case: return last attempt
    return out


def normalize_document_input(doc: list[bytes]) -> list[tuple[str, bytes]]:
    """
    Images sent to every agent, as (mime, bytes), each compressed to JPEG under ~800 KB.
    Accepts 1 to MAX_IMAGES images.
    """
    if not isinstance(doc, list) or not doc or any(
        (not isinstance(b, (bytes, bytearray)) or len(b) == 0) for b in doc
    ):
        raise ExtractionError("Image input must be a non-empty list of non-empty bytes objects.")
    if len(doc) > MAX_IMAGES:
        raise ExtractionError(f"Received {len(doc)} images; max allowed is {MAX_IMAGES}.")
    return [("image/jpeg", compress_image_under_limit(bytes(b), target_bytes=800_000)) for b in doc]


def best_enum_match(raw_value: Any, enum_list: list[Any]) -> tuple[Any, float]:
    """
    Returns: (best_enum_value, score_0_to_1)
    """
    if raw_value is None:
        return None, 0.0

    # Exact match (type-sensitive)
    for v in enum_list:
        if raw_value == v:
            return v, 1.0

    raw_str = str(raw_value).strip()
    if raw_str == "":
        return raw_value, 0.0

    # Match against stringified enum choices
    choices = [str(v) for v in enum_list]

    # If raw_str exactly matches a stringified enum, map to that enum value.
    try:
        idx = choices.index(raw_str)
        return enum_list[idx], 1.0
    except ValueError:
        pass

    best = process.extractOne(raw_str, choices, scorer=fuzz.WRatio)
    if not best:
        return raw_value, 0.0
    best_choice, best_score, _ = best
    score01 = float(best_score) / 100.0

    # Map back to original enum value with that string representation (first match)
    for i, s in enumerate(choices):
        if s == best_choice:
            return enum_list[i], score01
    return best_choice, score01


def best_enum_matches(raw_value: Any, enum_list: list[Any], top_k: int) -> list[tuple[Any, float]]:
    """
    Returns top-k fuzzy enum matches as (enum_value, score_0_to_1), sorted by score desc.
    """
    if top_k <= 0 or not enum_list:
        return []

    raw_str = str(raw_value).strip()
    if raw_str == "":
        return []

    choices = [str(v) for v in enum_list]
    hits = process.extract(raw_str, choices, scorer=fuzz.WRatio, limit=min(top_k, len(choices)))
    if not hits:
        return []

    out: list[tuple[Any, float]] = []
    seen: set[str] = set()
    for _, score, idx in hits:
        if not isinstance(idx, int) or idx < 0 or idx >= len(enum_list):
            continue
        enum_val = enum_list[idx]
        key = _canonical(enum_val)
        if key in seen:
            continue
        seen.add(key)
        out.append((enum_val, float(score) / 100.0))
    return out


def _apply_enum_rule_to_output(
    obj: Any,
    rule: EnumRule,
    *,
    path_prefix: tuple[str | int, ...] = (),
) -> None:
    """
    Mutates obj in-place: maps field to enum and adds <field>_% score.
    Handles "*" wildcard segments for arrays.
    """
    segments = rule.path

    def rec(current: Any, seg_idx: int, cur_path: tuple[str | int, ...]) -> None:
        if seg_idx >= len(segments):
            return
        seg = segments[seg_idx]
        last = seg_idx == len(segments) - 1

        if seg == "*":
            if isinstance(current, list):
                for i, item in enumerate(current):
                    rec(item, seg_idx + 1, cur_path + (i,))
            return

        # property segment
        if not isinstance(current, dict):
            return
        if seg not in current:
            return

        if last:
            raw_val = current.get(seg)
            mapped, score = best_enum_match(raw_val, rule.enum_list)
            current[seg] = mapped
            current[f"{seg}_%"] = score
            return

        rec(current.get(seg), seg_idx + 1, cur_path + (seg,))

    rec(obj, 0, path_prefix)


def apply_long_enum_remap(output_json: dict[str, Any], rules: list[EnumRule]) -> dict[str, Any]:
    """
    For each removed long enum: maps the generated value back into the original enum list
    and stores accuracy as <field>_%.
    """
    for r in rules:
        _apply_enum_rule_to_output(output_json, r)
    return output_json


def extract_enum_score_maps(obj: Any) -> dict[tuple[str | int, ...], float]:
    """
    Collects all keys ending in '_%' into a path->score map.
    Path points to the base field, e.g. ('sender','country') for 'country_%'.
    """
    scores: dict[tuple[str | int, ...], float] = {}

    def rec(node: Any, path: tuple[str | int, ...]) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.endswith("_%") and isinstance(v, (int, float)):
                    base = k[:-2]
                    scores[path + (base,)] = float(v)
            # Recurse into non-confidence keys
            for k, v in node.items():
                if isinstance(k, str) and k.endswith("_%"):
                    continue
                rec(v, path + (k,))
        elif isinstance(node, list):
            for i, it in enumerate(node):
                rec(it, path + (i,))

    rec(obj, ())
    return scores


def strip_confidence_keys(obj: Any) -> Any:
    """
    Deep-copies obj while removing keys ending in '_%'.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith("_%"):
                continue
            out[k] = strip_confidence_keys(v)
        return out
    if isinstance(obj, list):
        return [strip_confidence_keys(x) for x in obj]
    return obj


_MISSING = object()


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_for_compare(value: str) -> str:
    # Canonicalize unicode and collapse repeated whitespace.
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00A0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.casefold()


def _normalize_for_compare(obj: Any) -> Any:
    """
    Normalization used only for comparisons/aggregation:
      - strings: unicode normalize + collapse spaces + trim + casefold
      - numbers: near-integers (e.g. 10.0) -> 10
      - dict/list: recurse
    """
    if isinstance(obj, str):
        return _normalize_text_for_compare(obj)
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            rounded = round(obj)
            if abs(obj - rounded) <= 1e-9:
                return int(rounded)
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _normalize_for_compare(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_for_compare(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_for_compare(v) for v in obj]
    return obj


def _canonical(obj: Any) -> str:
    normalized = _normalize_for_compare(obj)
    try:
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(normalized)


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, dict)) and len(v) == 0:
        return True
    return False


def _mode_value(values: list[Any]) -> Any:
    """
    Picks the most frequent *non-empty* value if any exists, else most frequent empty value.
    Ties broken by first occurrence.
    """
    # Count by canonical representation for JSON-safe comparison
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for v in values:
        key = _canonical(v)
        if key not in buckets:
            buckets[key] = {"value": v, "count": 0, "empty": _is_empty_value(v)}
            order.append(key)
        buckets[key]["count"] += 1

    non_empty = [k for k in order if not buckets[k]["empty"]]
    empty = [k for k in order if buckets[k]["empty"]]

    def best(keys: list[str]) -> str | None:
        if not keys:
            return None
        max_count = max(buckets[k]["count"] for k in keys)
        for k in order:
            if k in keys and buckets[k]["count"] == max_count:
                return k
        return keys[0]

    best_non_empty = best(non_empty)
    if best_non_empty is not None:
        return buckets[best_non_empty]["value"]
    best_empty = best(empty)
    return buckets[best_empty]["value"] if best_empty is not None else None


def _frequency_of_value(chosen: Any, values: list[Any]) -> float:
    chosen_key = _canonical(chosen)
    hits = 0
    for v in values:
        if _canonical(v) == chosen_key:
            hits += 1
    return hits / float(len(values) or 1)


def _normalize_scalar_option_key(value: Any) -> str:
    normalized = _normalize_for_compare(value)
    if isinstance(normalized, str):
        return normalized
    try:
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(normalized)


def _format_scalar_option_label(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return "null"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value)


def _scalar_options(
    values: list[Any],
    *,
    source: Literal["AI", "enum"] = "AI",
) -> dict[str, dict[str, str | float]]:
    counts: dict[str, int] = {}
    total = float(len(values) or 1)
    for v in values:
        if _is_empty_value(v):
            continue
        key = _normalize_scalar_option_key(v)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return {}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        key: {
            "source": source,
            "accuracy": float(max(0.0, min(1.0, count / total))),
        }
        for key, count in ranked
    }


def _avg(nums: list[float]) -> float | None:
    if not nums:
        return None
    return sum(nums) / float(len(nums))


def _fuzzy_candidates_for_agent_value(
    raw_value: Any,
    rule: FuzzyEnumRule,
) -> dict[str, tuple[Any, float, Literal["AI", "enum"]]]:
    candidates: dict[str, tuple[Any, float, Literal["AI", "enum"]]] = {}
    enum_keys = {_canonical(v) for v in rule.enum_list}

    def add_candidate(value: Any, score: float, source: Literal["AI", "enum"]) -> None:
        key = _canonical(value)
        score = float(max(0.0, min(1.0, score)))
        prev = candidates.get(key)
        if prev is None:
            candidates[key] = (value, score, source)
            return

        prev_value, prev_score, prev_source = prev
        next_source: Literal["AI", "enum"] = "enum" if (prev_source == "enum" or source == "enum") else "AI"
        next_value = prev_value
        if prev_source != "enum" and source == "enum":
            # Prefer enum-provided representation when candidate belongs to the enum.
            next_value = value
        elif score > prev_score:
            next_value = value
        next_score = max(prev_score, score)
        candidates[key] = (next_value, next_score, next_source)

    # Keep model output as unconstrained candidate with full certainty.
    raw_source: Literal["AI", "enum"] = "enum" if _canonical(raw_value) in enum_keys else "AI"
    add_candidate(raw_value, 1.0, raw_source)
    for enum_value, score in best_enum_matches(raw_value, rule.enum_list, rule.top_k):
        add_candidate(enum_value, score, "enum")
    return candidates


def _aggregate_fuzzy_candidates(
    raw_values: list[Any],
    *,
    rule: FuzzyEnumRule,
    total_agents: int,
) -> tuple[Any, float, dict[str, dict[str, str | float]]]:
    summed_scores: dict[str, float] = {}
    value_by_key: dict[str, Any] = {}
    source_by_key: dict[str, Literal["AI", "enum"]] = {}
    order: list[str] = []

    for raw_value in raw_values:
        if raw_value is _MISSING:
            continue
        per_agent = _fuzzy_candidates_for_agent_value(raw_value, rule)
        for key, (value, score, source) in per_agent.items():
            if key not in value_by_key:
                value_by_key[key] = value
                source_by_key[key] = source
                order.append(key)
            elif source_by_key[key] != "enum" and source == "enum":
                # Prefer enum source and representation when available.
                source_by_key[key] = "enum"
                value_by_key[key] = value
            summed_scores[key] = summed_scores.get(key, 0.0) + float(score)

    if not summed_scores:
        return None, 0.0, {}

    denom = float(max(1, total_agents))
    avg_scores: dict[str, float] = {k: (v / denom) for k, v in summed_scores.items()}
    order_idx = {k: i for i, k in enumerate(order)}
    ranked = sorted(avg_scores.items(), key=lambda kv: (-kv[1], order_idx.get(kv[0], 0)))

    best_key, best_score = ranked[0]
    chosen = value_by_key.get(best_key)

    label_meta: dict[str, dict[str, str | float]] = {}
    for key, score in ranked:
        label = _format_scalar_option_label(value_by_key.get(key))
        if label == "":
            label = "''"
        score = float(max(0.0, min(1.0, score)))
        source = "enum" if source_by_key.get(key) == "enum" else "AI"
        prev = label_meta.get(label)
        if prev is None:
            label_meta[label] = {"source": source, "accuracy": score}
            continue

        prev_score = float(prev.get("accuracy", 0.0))
        prev_source = str(prev.get("source", "AI"))
        if score > prev_score or (score == prev_score and prev_source != "enum" and source == "enum"):
            label_meta[label] = {"source": source, "accuracy": score}

    ranked_opts = sorted(
        label_meta.items(),
        key=lambda kv: (-float(kv[1].get("accuracy", 0.0)), kv[0]),
    )
    if rule.top_k > 0:
        # Apply top-k independently per source so enum options are not starved by many AI-only variants.
        filtered_opts: list[tuple[str, dict[str, str | float]]] = []
        per_source_counts: dict[str, int] = {"AI": 0, "enum": 0}
        for label, meta in ranked_opts:
            source_raw = str(meta.get("source", "AI"))
            source = "enum" if source_raw == "enum" else "AI"
            if per_source_counts[source] >= rule.top_k:
                continue
            filtered_opts.append((label, meta))
            per_source_counts[source] += 1
            if per_source_counts["AI"] >= rule.top_k and per_source_counts["enum"] >= rule.top_k:
                break
        ranked_opts = filtered_opts
    opts = dict(ranked_opts)
    return chosen, float(max(0.0, min(1.0, best_score))), opts


def _path_matches_rule_path(
    concrete_path: tuple[str | int, ...],
    rule_path: tuple[str | Literal["*"], ...],
) -> bool:
    if len(concrete_path) != len(rule_path):
        return False
    for concrete_seg, rule_seg in zip(concrete_path, rule_path):
        if rule_seg == "*":
            if concrete_seg == "*" or isinstance(concrete_seg, int):
                continue
            return False
        if not isinstance(concrete_seg, str):
            return False
        if concrete_seg != rule_seg:
            return False
    return True


def _choose_match_key_for_dict_list(all_items: list[dict[str, Any]]) -> str | None:
    """
    Heuristic for aligning list[dict] items across agents.
    """
    if not all_items:
        return None

    # Candidate keys (exclude confidence keys)
    keys: set[str] = set()
    for it in all_items:
        for k in it.keys():
            if isinstance(k, str) and not k.endswith("_%"):
                keys.add(k)

    if not keys:
        return None

    total = len(all_items)

    preferred = {"id", "line_id", "lineId", "lineNumber", "sku", "code", "name", "description", "item", "title"}

    best_key = None
    best_score = 0.0
    for k in keys:
        vals: list[str] = []
        present = 0
        for it in all_items:
            v = it.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
                if s:
                    present += 1
                    vals.append(s.lower())
            elif isinstance(v, (int, float)):
                present += 1
                vals.append(str(v))

        if present == 0:
            continue

        coverage = present / float(total)
        uniq = len(set(vals)) / float(len(vals) or 1)
        score = coverage * uniq

        if k in preferred:
            score *= 1.15

        if score > best_score:
            best_score = score
            best_key = k

    # Must be reasonably usable
    if best_key is None or best_score < 0.30:
        return None
    return best_key


def merge_outputs_with_confidence(
    agent_outputs: list[dict[str, Any]],
    *,
    fuzzy_enum_rules: list[FuzzyEnumRule] | None = None,
    enforced_enum_paths: list[tuple[str | Literal["*"], ...]] | None = None,
    emit_scalar_options: bool = True,
) -> dict[str, Any]:
    """
    Merges N outputs into one JSON, adding <field>_% confidence everywhere.
    Confidence definition:
      - scalar fields: frequency_of_selected_value among agents,
        multiplied by average long-enum remap score among agents that selected the final value (if present).
      - dict/list fields: presence frequency (how many agents provided a non-empty value for that field).
      - enum_postprocess=fuzzy_refine(_k) scalar fields: average candidate probabilities
        from [LLM raw value @1.0 + top-k fuzzy enum matches], with details in <field>_opts.

    Notes:
      - This function expects agent_outputs may already contain *_% keys from enum remap.
      - It overwrites/normalizes *_% keys to final confidence values.
    """
    if not agent_outputs:
        return {}

    n = len(agent_outputs)
    fuzzy_rule_map: dict[tuple[str | Literal["*"], ...], FuzzyEnumRule] = {
        r.path: r for r in (fuzzy_enum_rules or [])
    }
    enum_paths = enforced_enum_paths or []

    enum_score_maps = [extract_enum_score_maps(o) for o in agent_outputs]
    cleaned = [strip_confidence_keys(o) for o in agent_outputs]

    def rec(
        per_agent_values: list[Any],
        path: tuple[str | int, ...],
    ) -> Any:
        # Normalize missing
        vals = [v if v is not _MISSING else None for v in per_agent_values]

        # Dict merge
        if any(isinstance(v, dict) for v in vals if v is not None):
            # Collect keys
            keys: set[str] = set()
            dicts = [v for v in vals if isinstance(v, dict)]
            for d in dicts:
                keys.update(str(k) for k in d.keys())

            out: dict[str, Any] = {}
            for k in sorted(keys):
                child_vals = []
                for v in vals:
                    if isinstance(v, dict) and k in v:
                        child = v.get(k)
                        child_vals.append(child)
                    else:
                        child_vals.append(_MISSING)

                field_path = path + (k,)
                fuzzy_rule = fuzzy_rule_map.get(field_path)
                if fuzzy_rule and not any(
                    isinstance(cv, (dict, list)) for cv in child_vals if cv is not _MISSING
                ):
                    chosen, conf, opts = _aggregate_fuzzy_candidates(
                        child_vals,
                        rule=fuzzy_rule,
                        total_agents=n,
                    )
                    out[k] = chosen
                    out[f"{k}_%"] = float(max(0.0, min(1.0, conf)))
                    if opts:
                        out[f"{k}_opts"] = dict(opts)
                    continue

                merged_child = rec(child_vals, path + (k,))
                out[k] = merged_child

                # Presence confidence for containers is handled at parent key;
                # here we set confidence for each key based on its child.
                # If child is scalar -> frequency-based, else presence-based.
                if isinstance(merged_child, (dict, list)):
                    conf = sum(
                        1 for v in child_vals if (v is not _MISSING and not _is_empty_value(v))
                    ) / float(n)
                else:
                    # scalar: compute mode frequency, and adjust by enum remap score if available
                    chosen = merged_child
                    # values used for freq: per-agent scalar (coerce missing to None)
                    scalar_vals = [(cv if cv is not _MISSING else None) for cv in child_vals]
                    freq = _frequency_of_value(chosen, scalar_vals)

                    # enum score: only for agents whose value == chosen and have a score at this path
                    score_hits: list[float] = []
                    for i in range(n):
                        av = scalar_vals[i]
                        if _canonical(av) != _canonical(chosen):
                            continue
                        s = enum_score_maps[i].get(path + (k,))
                        if s is not None:
                            score_hits.append(float(s))
                    enum_score = _avg(score_hits)
                    conf = freq * enum_score if enum_score is not None else freq

                out[f"{k}_%"] = float(max(0.0, min(1.0, conf)))
                if emit_scalar_options and not isinstance(merged_child, (dict, list)):
                    source: Literal["AI", "enum"] = (
                        "enum" if any(_path_matches_rule_path(field_path, rp) for rp in enum_paths) else "AI"
                    )
                    opts = _scalar_options(scalar_vals, source=source)
                    if opts:
                        out[f"{k}_opts"] = opts

            return out

        # List merge
        if any(isinstance(v, list) for v in vals if v is not None):
            lists = [v for v in vals if isinstance(v, list)]
            # Presence confidence for the list field itself is handled at parent key.
            # Here we merge list content. If it's list[dict], attempt alignment+union.
            # Else pick best representative list (mode of non-empty lists).
            if lists and all((len(l) == 0 or isinstance(l[0], dict)) for l in lists):
                # Flatten items for match-key selection
                all_items: list[dict[str, Any]] = []
                for l in lists:
                    for it in l:
                        if isinstance(it, dict):
                            all_items.append(it)
                match_key = _choose_match_key_for_dict_list(all_items)

                # Build per-agent index by id (if match_key usable), else align by index.
                if match_key:
                    # union of ids (stable order: first agent then others)
                    union_ids: list[str] = []
                    seen: set[str] = set()

                    def add_ids_from(lst: list[dict[str, Any]]) -> None:
                        for it in lst:
                            if not isinstance(it, dict):
                                continue
                            raw = it.get(match_key)
                            if raw is None:
                                continue
                            sid = str(raw).strip()
                            if not sid:
                                continue
                            if sid not in seen:
                                seen.add(sid)
                                union_ids.append(sid)

                    # prefer first agent ordering if possible
                    for v in vals:
                        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                            add_ids_from(v)  # type: ignore[arg-type]
                            break
                    # add remaining ids
                    for v in vals:
                        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                            add_ids_from(v)  # type: ignore[arg-type]

                    # per-agent id->item mapping
                    per_agent_maps: list[dict[str, dict[str, Any]]] = []
                    for v in vals:
                        m: dict[str, dict[str, Any]] = {}
                        if isinstance(v, list):
                            for it in v:
                                if not isinstance(it, dict):
                                    continue
                                raw = it.get(match_key)
                                if raw is None:
                                    continue
                                sid = str(raw).strip()
                                if sid and sid not in m:
                                    m[sid] = it
                        per_agent_maps.append(m)

                    merged_items: list[Any] = []
                    for sid in union_ids:
                        child_vals = []
                        for i in range(n):
                            it = per_agent_maps[i].get(sid)
                            child_vals.append(it if it is not None else _MISSING)
                        merged_item = rec(child_vals, path + ("*",))  # wildcard-ish for nested scoring
                        merged_items.append(merged_item)
                    return merged_items

            # Fallback: pick representative list (mode of non-empty), no per-element confidence for scalars.
            # Keep it deterministic.
            rep = _mode_value([l for l in vals if isinstance(l, list)] or [[]])
            return rep if isinstance(rep, list) else []

        # Scalar merge
        chosen = _mode_value(vals)
        return chosen

    # Top-level is assumed dict-like; we return merged dict with confidence keys inside.
    merged = rec(cleaned, ())
    if not isinstance(merged, dict):
        # If schema top-level isn't object, still annotate with a wrapper.
        merged = {"value": merged, "value_%": 1.0}

    # Top-level confidence keys are already embedded per field.
    return merged


def annotate_single_run_confidence(output: dict[str, Any]) -> dict[str, Any]:
    """
    For N=1:
      - set every <field>_% to 1.0
      - except fields that already have <field>_% from enum remap: keep that value
    """
    def rec(node: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            # First copy existing keys (including existing *_%)
            for k, v in node.items():
                out[k] = rec(v, path + (k,)) if not (isinstance(k, str) and k.endswith("_%")) else v

            # Add missing confidence keys for every non-confidence field
            for k, v in list(out.items()):
                if not isinstance(k, str) or k.endswith("_%"):
                    continue
                conf_key = f"{k}_%"
                if conf_key not in out:
                    out[conf_key] = 1.0
            return out
        if isinstance(node, list):
            return [rec(x, path + (i,)) for i, x in enumerate(node)]
        return node

    return rec(output, ())


def _build_google_parts_from_media(
    *,
    types_module: Any,
    settings: ModelSettings,
    inline_images: list[tuple[str, bytes]],
) -> list[Any]:
    parts: list[Any] = [types_module.Part.from_bytes(data=b, mime_type=mime) for mime, b in inline_images]
    parts.append(types_module.Part.from_text(text=settings.user_prompt))
    return parts


def _build_google_structured_config(
    *,
    types_module: Any,
    settings: ModelSettings,
    schema: dict[str, Any],
) -> Any:
    cfg_kwargs: dict[str, Any] = {
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    if settings.system_prompt:
        cfg_kwargs["system_instruction"] = settings.system_prompt
    if settings.temperature is not None:
        cfg_kwargs["temperature"] = float(settings.temperature)
    if settings.top_p is not None:
        cfg_kwargs["top_p"] = float(settings.top_p)
    if settings.max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = int(settings.max_output_tokens)
    if settings.thinking_level:
        cfg_kwargs["thinking_config"] = types_module.ThinkingConfig(thinking_level=settings.thinking_level)
    return types_module.GenerateContentConfig(**cfg_kwargs)


async def _call_google_structured(
    *,
    settings: ModelSettings,
    parts: list[Any],
    config: Any,
    client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    One agent: one Gemini call with response_json_schema and application/json, retried on transient errors.
    """
    from google import genai
    from google.genai import types

    # API key can be GEMINI_API_KEY or GOOGLE_API_KEY per SDK docs.
    if client is None:
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    async def _invoke() -> tuple[dict[str, Any], dict[str, Any]]:
        resp = await client.aio.models.generate_content(
            model=settings.model,
            contents=[types.Content(role="user", parts=parts)],
            config=config,
        )
        text = getattr(resp, "text", "") or ""
        data = _safe_json_loads(text)

        usage = _normalize_google_usage(getattr(resp, "usage_metadata", None))
        usage["provider"] = "google"
        usage["model"] = settings.model
        usage["vendor_meta"] = _extract_google_vendor_meta(resp)
        return data, usage

    try:
        return await run_with_retries(
            _invoke,
            dependency="google",
            operation_name="structured",
            logger=logger,
            is_transient=is_transient_google_error,
            retry_policy=AI_PROVIDER_RETRY_POLICY,
        )
    except Exception as exc:
        raise ExtractionError(f"Google call failed: {exc}") from exc


def _safe_json_loads(text: str) -> dict[str, Any]:
    """
    Robust JSON parser for cases where the model returns extra text.
    """
    text = text.strip()
    if not text:
        return {}

    # If already clean JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass

    # Try to extract a JSON object from within code fences or surrounding text
    # Find first {...} block (naive but practical)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            pass

    raise ExtractionError("Model response was not valid JSON.")


def _normalize_google_usage(usage_md: Any) -> dict[str, Any]:
    if usage_md is None:
        return {
            "input_tokens": 0,
            "input_cached_tokens": 0,
            "input_uncached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "available": False,
            "cache_reporting_available": False,
        }

    # usage_metadata is dict-like/object-like depending SDK versions.
    def g(path: list[str], default: int = 0) -> int:
        cur: Any = usage_md
        for k in path:
            if isinstance(cur, dict):
                cur = cur.get(k, None)
            else:
                cur = getattr(cur, k, None)
            if cur is None:
                return default
        if isinstance(cur, (int, float)):
            return int(cur)
        return default

    inp = g(["prompt_token_count"], 0)
    out = g(["candidates_token_count"], 0)
    total = g(["total_token_count"], inp + out)

    cached_candidates = [
        g(["cached_content_token_count"], -1),
        g(["cache_read_tokens"], -1),
    ]
    cached = next((v for v in cached_candidates if v >= 0), 0)
    uncached = max(0, inp - cached)
    cache_reporting_available = any(v >= 0 for v in cached_candidates)

    return {
        "input_tokens": inp,
        "input_cached_tokens": cached,
        "input_uncached_tokens": uncached,
        "output_tokens": out,
        "total_tokens": total,
        "available": True,
        "cache_reporting_available": cache_reporting_available,
    }


def _extract_google_vendor_meta(resp: Any) -> dict[str, str]:
    """Model version, response ID and finish reason of one Gemini call, for troubleshooting."""
    meta: dict[str, str] = {}
    if resp is None:
        return meta
    for attr in ("model_version", "response_id"):
        val = getattr(resp, attr, None)
        if isinstance(val, str) and val.strip():
            meta[attr] = val.strip()
    candidates = getattr(resp, "candidates", None)
    if candidates and len(candidates) > 0:
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None:
            meta["finish_reason"] = str(finish_reason)
    return meta


async def extract_parallel(
    *,
    document: list[bytes],
    schema: dict[str, Any],
    settings: ModelSettings,
    n_agents: int,
    provider_client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Main pipeline:
      1) Validate schema
      2) Relax enum constraints:
         - remove long enums (>50) -> enum_rules (post-hoc remap)
         - remove enums marked with enum_postprocess=fuzzy_refine(_k) -> fuzzy_enum_rules
      3) Normalize input images (<= 20 images, JPEG under ~800 KB each)
      4) Run N agents concurrently (same inputs)
      5) Remap removed long enums back into original enum values, storing per-field accuracy as <field>_%
      6) Merge agent outputs recursively and attach <field>_% confidence everywhere
         (plus <field>_opts with the values proposed by the agents)
      7) Return (final_json, usage_json)
    """
    if n_agents <= 0:
        raise ExtractionError("n_agents must be >= 1")
    if settings.provider != "google":
        raise ExtractionError(f"Unsupported provider: {settings.provider}")

    normalized_schema = normalize_extraction_schema(schema)
    validate_json_schema(normalized_schema)
    relaxed_schema, enum_rules, fuzzy_enum_rules = relax_schema_enums(normalized_schema)
    enforced_enum_paths = collect_enforced_enum_paths(relaxed_schema)

    # Prepare images, prompt and config once and reuse them across agent calls.
    from google.genai import types

    inline_images = normalize_document_input(document)
    parts = _build_google_parts_from_media(types_module=types, settings=settings, inline_images=inline_images)
    config = _build_google_structured_config(types_module=types, settings=settings, schema=relaxed_schema)

    # Run agents
    async def run_one() -> tuple[dict[str, Any], dict[str, Any]]:
        data, usage = await _call_google_structured(
            settings=settings, parts=parts, config=config, client=provider_client,
        )
        # Post-hoc enum remap for removed long enums (adds <field>_% match accuracy keys)
        if enum_rules:
            data = apply_long_enum_remap(data, enum_rules)
        return data, usage

    results = await asyncio.gather(*[run_one() for _ in range(n_agents)], return_exceptions=True)

    outputs: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    errors: list[str] = []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
            continue
        out, usage = r
        if not isinstance(out, dict):
            out = {"value": out}
        outputs.append(out)
        usages.append(usage)

    if not outputs:
        raise ExtractionError(f"All agents failed. Errors: {errors}")

    # Merge
    if len(outputs) == 1:
        if fuzzy_enum_rules:
            # Use merge path to apply fuzzy enum-postprocessing even for single-agent runs.
            final = merge_outputs_with_confidence(
                outputs,
                fuzzy_enum_rules=fuzzy_enum_rules,
                enforced_enum_paths=enforced_enum_paths,
                emit_scalar_options=False,
            )
        else:
            final = annotate_single_run_confidence(outputs[0])
    else:
        final = merge_outputs_with_confidence(
            outputs,
            fuzzy_enum_rules=fuzzy_enum_rules,
            enforced_enum_paths=enforced_enum_paths,
        )

    usage_summary = {
        "provider": settings.provider,
        "model": settings.model,
        "agents_requested": n_agents,
        "agents_succeeded": len(outputs),
        "agents_failed": len(results) - len(outputs),
        "errors": errors,
        "totals": {
            "input_tokens": sum(u.get("input_tokens", 0) or 0 for u in usages),
            "input_cached_tokens": sum(u.get("input_cached_tokens", 0) or 0 for u in usages),
            "input_uncached_tokens": sum(u.get("input_uncached_tokens", 0) or 0 for u in usages),
            "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in usages),
            "total_tokens": sum(u.get("total_tokens", 0) or 0 for u in usages),
        },
        "by_agent": usages,
    }
    return final, usage_summary
