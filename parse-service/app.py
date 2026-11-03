"""
Parse service for Odoo: reads the label of a raw-material container from a photo and returns JSON.

Pipeline
    1. Reading:   N AI agents read the label in parallel (extract_parallel in scraper_agent.py);
                  their answers are merged field by field, with an agreement score for each field.
    2. Matching:  rapidfuzz recognises the supplier and shortlists the closest products of the catalog.
    3. Choice:    a second AI call, with the photo and the shortlist, picks one product code or "none".

The service is stateless: it never connects to Odoo. Odoo sends the photo, the label schema and the
product catalog with every request, and writes the result itself.
"""
import base64
import binascii
import logging
import os
import re
import time
import unicodedata
from typing import Any

from fastapi import FastAPI, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel
from rapidfuzz import fuzz, process, utils

from scraper_agent import ExtractionError, ModelSettings, extract_parallel

# Configuration, read from the environment (.env file passed to the container by compose.yaml)
MODEL = os.environ.get("PARSE_MODEL", "gemini-3.5-flash-lite")
N_AGENTS = int(os.environ.get("PARSE_N_AGENTS", "3"))
TIMEOUT_SEC = int(os.environ.get("PARSE_TIMEOUT_SEC", "60"))   # max duration of each Gemini call

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("parse")
logging.getLogger("httpx").setLevel(logging.WARNING)            # no log line for every HTTP call

# One Gemini client for the whole service. HttpOptions.timeout is in milliseconds:
# without it a slow call can hang for minutes.
GEMINI_CLIENT = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(timeout=TIMEOUT_SEC * 1000),
)

# Matching thresholds, as similarity scores from 0 (nothing in common) to 100 (identical)
N_CANDIDATES = 5            # products passed to the AI choice
SUPPLIER_THRESHOLD = 80     # below this score the supplier read on the label is not matched
CANDIDATE_THRESHOLD = 60    # among the supplier's products, below this score nothing is proposed
LEGAL_FORMS = {"SPA", "SRL", "SNC", "SAS", "SA", "AG", "SE", "GMBH", "LTD", "INC"}   # ignored in supplier names

READING_SYSTEM_PROMPT = (
    "You work at the goods-receipt desk of a cosmetics laboratory. "
    "Read the label of the container in the photo and fill in the requested fields. "
    "Rules: "
    "1) Report only what is printed and legible in the photo: do not deduce, complete or guess. "
    "2) If a value is missing or illegible, leave text empty and use null for numbers: "
    "an empty field is a correct answer, an invented value is a serious error. "
    "3) Read the expiry date only next to wording such as EXP, EXPIRY, USE BY, BEST BEFORE, SCAD., SCADENZA, "
    "DA CONSUMARSI. Ignore production dates (PROD., MFG, MFD), analysis or packing dates and lot digits. "
    "4) The quantity is only the NET weight or volume, never gross weight or tare."
)
READING_USER_PROMPT = "Read the label of the container in this photo."

NONE_CODE = "none"          # value the AI picks when no candidate is the product in the photo
CHOICE_SYSTEM_PROMPT = (
    "You work at the goods-receipt desk of a cosmetics laboratory. "
    "Match the container in the photo to a product of the warehouse catalog. "
    "A similar name is not enough: it must be the same substance or the same component. "
    f"If no candidate matches, choose '{NONE_CODE}'."
)

app = FastAPI(title="Parse - container label reader")


class Product(BaseModel):
    """One catalog entry, as sent by Odoo."""
    code: str
    name: str
    suppliers: list[str] = []


class ReadRequest(BaseModel):
    """JSON body sent by the Odoo module."""
    photo: str                          # base64-encoded image
    label_schema: dict[str, Any]        # JSON Schema of the label fields (schemas/label.json in the module)
    catalog: list[Product] = []         # empty = only step 1 (reading) is run


def text(reading: dict[str, Any], field: str) -> str:
    """Text value of a field read on the label; empty string if missing or not text."""
    value = reading.get(field)
    return value.strip() if isinstance(value, str) else ""


def normalize_supplier(name: str) -> str:
    """Make supplier names comparable: 'A.B.C. S.p.A.' and 'ABC' both become 'ABC'."""
    # NFKD splits letters from accents; encoding to ASCII then drops the accents
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.upper().replace(".", "")
    words = re.sub(r"[^0-9A-Z]+", " ", name).split()
    return " ".join(word for word in words if word not in LEGAL_FORMS)


def match_supplier(read_name: str, catalog: list[Product]) -> dict[str, Any]:
    """Match the supplier read on the label to one of the suppliers in the catalog."""
    names = sorted({supplier for product in catalog for supplier in product.suppliers})
    if not read_name or not names:
        return {"read": read_name, "name": None, "score": 0.0}
    # WRatio copes with case, word order and partial names; extractOne returns (value, score, index) or None
    found = process.extractOne(read_name, names, scorer=fuzz.WRatio, processor=normalize_supplier)
    if found is None or found[1] < SUPPLIER_THRESHOLD:
        return {"read": read_name, "name": None, "score": round(found[1], 1) if found else 0.0}
    return {"read": read_name, "name": found[0], "score": round(found[1], 1)}


def best_matches(queries: list[str], group: list[Product]) -> list[tuple[Product, float]]:
    """Best score of each product of the group over all queries, sorted from the highest."""
    names = [product.name for product in group]
    scores: dict[int, float] = {}
    for query in queries:
        for _, score, index in process.extract(
            query, names, scorer=fuzz.WRatio, processor=utils.default_process, limit=N_CANDIDATES
        ):
            scores[index] = max(scores.get(index, 0.0), score)
    return sorted(((group[i], s) for i, s in scores.items()), key=lambda pair: pair[1], reverse=True)


def shortlist(reading: dict[str, Any], catalog: list[Product], supplier: str | None) -> list[dict[str, Any]]:
    """Step 2: up to N_CANDIDATES products, first from the recognised supplier, then from the whole catalog."""
    queries = [q for q in (text(reading, "product_name"), text(reading, "printed_name")) if q]
    if not queries or not catalog:
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(product: Product, score: float, from_supplier: bool) -> None:
        candidates.append({
            "code": product.code,
            "name": product.name,
            "suppliers": product.suppliers,
            "score": round(score, 1),
            "from_supplier": from_supplier,
        })
        seen.add(product.code)

    if supplier:
        supplier_products = [p for p in catalog if supplier in p.suppliers]
        for product, score in best_matches(queries, supplier_products):
            if score >= CANDIDATE_THRESHOLD and len(candidates) < N_CANDIDATES:
                add(product, score, True)

    for product, score in best_matches(queries, catalog):
        if len(candidates) >= N_CANDIDATES:
            break
        if product.code not in seen:
            add(product, score, False)

    return candidates


async def choose_product(photo: bytes, reading: dict[str, Any], supplier: dict[str, Any],
                         candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Step 3: the AI looks at the photo again and picks one candidate code, or 'none'."""
    if not candidates:
        return {"code": None, "name": None, "confidence": None, "reason": "no candidates"}, {}

    # Schema built on the fly: the enum holds ONLY the candidate codes plus "none",
    # so the AI cannot invent a code that is not in the catalog.
    choice_schema = {
        "type": "object",
        "title": "product_choice",
        "additionalProperties": False,
        "required": ["code", "reason"],
        "properties": {
            "code": {
                "type": "string",
                "enum": [c["code"] for c in candidates] + [NONE_CODE],
                "description": f"Code of the product that is the same product as the photo, or '{NONE_CODE}'.",
            },
            "reason": {"type": "string", "description": "One short sentence explaining the choice."},
        },
    }
    listing = "\n".join(
        f"- {c['code']}: {c['name']} (suppliers: {', '.join(c['suppliers']) or 'none'})" for c in candidates
    )
    prompt = (
        f"The label reads the product \"{text(reading, 'printed_name')}\" "
        f"from the supplier \"{supplier['read']}\".\n"
        f"Candidate products from the catalog:\n{listing}\n"
        f"Look at the photo and choose the matching product code, or '{NONE_CODE}'."
    )
    settings = ModelSettings(provider="google", model=MODEL,
                             system_prompt=CHOICE_SYSTEM_PROMPT, user_prompt=prompt)
    result, usage = await extract_parallel(
        document=[photo], schema=choice_schema, settings=settings,
        n_agents=N_AGENTS, provider_client=GEMINI_CLIENT,
    )
    code = result.get("code")
    names = {c["code"]: c["name"] for c in candidates}
    choice = {
        "code": code if code in names else None,      # "none" becomes None
        "name": names.get(code),
        "confidence": result.get("code_%"),           # share of agents agreeing on the code
        "reason": result.get("reason", ""),
    }
    return choice, usage


@app.get("/health")
def health() -> dict[str, Any]:
    # the API key is never returned
    return {"status": "ok", "model": MODEL, "agents": N_AGENTS, "timeout_sec": TIMEOUT_SEC}


@app.post("/read")
async def read(request: ReadRequest) -> dict[str, Any]:
    try:
        photo = base64.b64decode(request.photo, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="photo: invalid base64")
    log.info("request: photo %d KB, catalog %d products", len(photo) // 1024, len(request.catalog))

    # Step 1: reading. document=[photo] is a list of images; raw bytes alone would be treated as a PDF.
    start = time.monotonic()
    settings = ModelSettings(provider="google", model=MODEL,
                             system_prompt=READING_SYSTEM_PROMPT, user_prompt=READING_USER_PROMPT)
    try:
        reading, usage = await extract_parallel(
            document=[photo], schema=request.label_schema, settings=settings,
            n_agents=N_AGENTS, provider_client=GEMINI_CLIENT,
        )
    except ExtractionError as error:
        log.error("step 1 failed after %.1f s: %s", time.monotonic() - start, error)
        raise HTTPException(status_code=502, detail=str(error))     # 502 = upstream (Gemini) error
    log.info("step 1: %.1f s, agents succeeded %s/%d, lot %s",
             time.monotonic() - start, usage.get("agents_succeeded"), N_AGENTS, reading.get("lot"))

    # Step 2: supplier and candidate products
    supplier = match_supplier(text(reading, "supplier"), request.catalog)
    candidates = shortlist(reading, request.catalog, supplier["name"])
    log.info("step 2: supplier %s, first candidate %s", supplier["name"],
             f"{candidates[0]['code']} {candidates[0]['name']}" if candidates else "none")

    # Step 3: AI choice among the candidates
    start = time.monotonic()
    try:
        choice, choice_usage = await choose_product(photo, reading, supplier, candidates)
    except ExtractionError as error:
        log.error("step 3 failed after %.1f s: %s", time.monotonic() - start, error)
        raise HTTPException(status_code=502, detail=str(error))
    log.info("step 3: %.1f s, chose %s (confidence %s)", time.monotonic() - start, choice["code"], choice["confidence"])

    return {
        "reading": reading,
        "supplier": supplier,
        "candidates": candidates,
        "choice": choice,
        "usage": {"reading": usage, "choice": choice_usage},
    }
