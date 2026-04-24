"""
Client lookup for Hoffmann purchase orders.

Strategy (strict, in this order):
1. Extract the CIF/VAT/NIF of the CUSTOMER (never Hoffmann) from the PO.
2. Look it up in the clients table. If NOT found → error.
3. If found with only one row → that is the client.
4. If found with several rows (same CIF, multiple delivery addresses) →
   filter by postal code extracted from the PO delivery address.
5. If still several candidates with the same CIF+postal code →
   fuzzy-match their 'Direccion envio' against the PO delivery address
   and pick the one with >= 90% similarity. If none ≥ 90% → error.
"""

from __future__ import annotations
import re
import pandas as pd
from rapidfuzz import fuzz, process


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_clients(path: str) -> list[dict]:
    """
    Load clients Excel with columns:
      VAT/NIF/CIF | Codigo Postal | Direccion envio | Numero clinete | Pais | Codigos Propios
    Returns list of dicts with normalized keys.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if ("vat" in cl or "nif" in cl or "cif" in cl) and "vat/nif/cif" in cl.replace(" ", ""):
            col_map["vat"] = c
        elif col_map.get("vat") is None and ("vat" in cl or "nif" in cl or "cif" in cl):
            col_map["vat"] = c
        elif "codigo" in cl and "postal" in cl:
            col_map["postal"] = c
        elif "direcc" in cl and "envio" in cl:
            col_map["address"] = c
        elif "numero" in cl and ("clinete" in cl or "cliente" in cl):
            col_map["number"] = c
        elif cl in ("pais", "país") or cl.startswith("pais") or cl.startswith("país"):
            col_map["country"] = c
        elif "codigos" in cl and "propr" in cl:
            col_map["own_codes"] = c

    clients = []
    for _, row in df.iterrows():
        vat = str(row.get(col_map.get("vat", ""), "")).strip().upper()
        postal = str(row.get(col_map.get("postal", ""), "")).strip()
        addr = str(row.get(col_map.get("address", ""), "")).strip()
        num = str(row.get(col_map.get("number", ""), "")).strip()
        country = str(row.get(col_map.get("country", ""), "")).strip().upper()
        own_codes = str(row.get(col_map.get("own_codes", ""), "")).strip() if col_map.get("own_codes") else ""

        if not vat or not num:
            continue

        clients.append({
            "vat": _normalize_vat(vat),
            "postal_code": postal,
            "address": addr,
            "client_number": num,
            "country": country,
            "own_codes": own_codes,
        })
    return clients


# ─── Helpers ──────────────────────────────────────────────────────────────────

# Hoffmann's own VAT numbers — never use these as CUSTOMER identifiers.
# (Aludium, EFAPEL, Talgo, TE, etc. are CUSTOMERS — they are NOT in this list.)
HOFFMANN_VATS = {
    # Hoffmann Iberia Quality Tools SL (Spain)
    "B85500882", "ESB85500882",
    # Hoffmann Iberia Quality Tools SL - Sucursal em Portugal
    "980671566", "PT980671566",
}


def _normalize_vat(vat: str) -> str:
    """Strip all whitespace and uppercase. Keep country prefix if present."""
    if not vat:
        return ""
    return re.sub(r"\s+", "", vat).upper()


def extract_vat_from_text(text: str) -> list[str]:
    """
    Extract candidate VAT/NIF/CIF numbers from PDF text.
    Returns a list of candidates (excluding Hoffmann's known VATs).
    """
    if not text:
        return []

    candidates = []

    # Pattern 1: explicit labels. Allow optional newlines/spaces between label and value.
    label_pat = re.compile(
        r'(?:VAT(?:/NIF)?|NIF|CIF|Tax\s*ID|Tax\s*number|C\.I\.F\.?|V\.A\.T\.?|N\.I\.F\.?)'
        r'\s*[:/\-]*\s*'
        r'([A-Z]{0,3}\s?[A-Z0-9][A-Z0-9\-]{5,14})',
        re.IGNORECASE,
    )
    for m in label_pat.finditer(text):
        v = _normalize_vat(m.group(1))
        if 6 <= len(v) <= 15:  # sanity check
            candidates.append(v)

    # Pattern 2: raw VAT-looking strings with country prefix (more permissive)
    raw_pat = re.compile(r'\b((?:ES|PT|FR|DE|IT|UK|GB|NL|BE|LU|IE|AT|CH)[A-Z0-9]{7,12})\b')
    for m in raw_pat.finditer(text):
        candidates.append(_normalize_vat(m.group(1)))

    # Deduplicate preserving order
    seen = set()
    unique = []
    for v in candidates:
        if v and v not in seen:
            seen.add(v)
            unique.append(v)

    # Remove Hoffmann's own VATs
    unique = [v for v in unique if not _is_hoffmann_vat(v)]
    return unique


def _is_hoffmann_vat(vat: str) -> bool:
    """Check if a VAT looks like one of Hoffmann's known numbers."""
    if not vat:
        return False
    v = _normalize_vat(vat)
    for hof in HOFFMANN_VATS:
        if v == hof:
            return True
    return False


def extract_postal_code(text: str) -> str:
    """
    Extract a postal code from text. Supports:
    - Portuguese with dash: 7005-838
    - Spanish 5-digit: 48340
    - European 4-5 digit
    """
    if not text:
        return ""
    # Portuguese format first (4 digits - 3 digits)
    m = re.search(r'\b(\d{4}-\d{3})\b', text)
    if m:
        return m.group(1)
    # Spanish/European 4-5 digit
    m = re.search(r'\b(\d{4,5})\b', text)
    if m:
        return m.group(1)
    return ""


# ─── Main client lookup ───────────────────────────────────────────────────────

def find_client(
    customer_vat: str,
    delivery_address: str,
    clients: list[dict],
    raw_text_fallback: str = "",
) -> dict:
    """
    Find the client matching the PO.

    Returns a dict with:
      - client_number, country, address (when found)
      - error (str) if not found/ambiguous
    """
    empty_error = {
        "client_number": "",
        "country": "",
        "address": "",
        "error": "",
    }

    if not clients:
        empty_error["error"] = "No clients table loaded"
        return empty_error

    # ── Step 1: Get VAT candidate(s) ───────────────────────────────────
    vat_candidates = []
    if customer_vat:
        v = _normalize_vat(customer_vat)
        if v and not _is_hoffmann_vat(v):
            vat_candidates.append(v)

    # If the LLM did not provide a VAT, try to find it in the raw PDF text
    if not vat_candidates and raw_text_fallback:
        vat_candidates = extract_vat_from_text(raw_text_fallback)

    if not vat_candidates:
        empty_error["error"] = "No customer VAT/NIF/CIF could be extracted from the PDF"
        return empty_error

    # ── Step 2: Find rows that match ANY of the candidate VATs ─────────
    matching = []
    matched_vat = ""
    for candidate in vat_candidates:
        rows = [c for c in clients if c["vat"] == candidate]
        if rows:
            matching = rows
            matched_vat = candidate
            break

    # Also try stripping country prefix (e.g. "ESB84528553" → "B84528553")
    if not matching:
        for candidate in vat_candidates:
            stripped = re.sub(r'^(ES|PT|FR|DE|IT|UK|GB|NL)', '', candidate)
            rows = [c for c in clients if c["vat"] == stripped or c["vat"].endswith(stripped)]
            if rows:
                matching = rows
                matched_vat = candidate
                break

    if not matching:
        empty_error["error"] = (
            f"Customer VAT/NIF/CIF not found in clients table "
            f"(tried: {', '.join(vat_candidates[:5])})"
        )
        return empty_error

    # ── Step 3: Single match → done ────────────────────────────────────
    if len(matching) == 1:
        c = matching[0]
        return {
            "client_number": c["client_number"],
            "country": c["country"],
            "address": c["address"],
            "error": "",
        }

    # ── Step 4: Multiple rows with same VAT → filter by postal code ────
    pdf_postal = extract_postal_code(delivery_address)
    if pdf_postal:
        by_postal = [c for c in matching if c["postal_code"] == pdf_postal]
        if len(by_postal) == 1:
            c = by_postal[0]
            return {
                "client_number": c["client_number"],
                "country": c["country"],
                "address": c["address"],
                "error": "",
            }
        if by_postal:
            matching = by_postal  # narrow down for step 5

    # ── Step 5: Still multiple → fuzzy match addresses, >= 90% ─────────
    if delivery_address:
        addresses = [c["address"] for c in matching]
        result = process.extractOne(
            delivery_address,
            addresses,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90,
        )
        if result:
            best_addr = result[0]
            best = next(c for c in matching if c["address"] == best_addr)
            return {
                "client_number": best["client_number"],
                "country": best["country"],
                "address": best["address"],
                "error": "",
            }

    # If we got here: VAT found but cannot disambiguate the specific address
    empty_error["error"] = (
        f"Customer VAT '{matched_vat}' matches {len(matching)} rows but the "
        f"delivery address could not be matched with >= 90% similarity"
    )
    return empty_error
