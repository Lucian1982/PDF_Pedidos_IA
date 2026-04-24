"""
Hoffmann PDF extractor.

Strategy:
1. Extract text from PDF with pdfplumber.
2. Send the text to OpenAI. Always.
3. Normalize the PO date to DD/MM/YYYY (this is always the value sent in the HEAD).
4. For each line with an empty hoffmannArticle, fuzzy-match the line description
   against the catalog's product names (Kurzbeschreibung + Marke + Produktname).
   If confidence >= 90%, use the catalog reference.
5. Validate: orderNumber + every line must have ref+qty+price.
6. If all good → pipe-delimited HEAD|LINE|... (HTTP 200).
   Otherwise → JSON error with context (HTTP 422).
"""

from __future__ import annotations
import re
import pdfplumber

from .llm import extract_with_llm
from .clients import find_client
from .catalog import find_reference_by_name, build_ref_index, resolve_customer_part_number


# ─── Helpers ──────────────────────────────────────────────────────────────────

_MONTHS = {
    # Spanish
    "ene": 1, "enero": 1, "feb": 2, "febrero": 2, "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4, "may": 5, "mayo": 5, "jun": 6, "junio": 6,
    "jul": 7, "julio": 7, "ago": 8, "agosto": 8, "sep": 9, "sept": 9,
    "septiembre": 9, "oct": 10, "octubre": 10, "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
    # English
    "jan": 1, "january": 1, "feb": 2, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    # Portuguese
    "jan": 1, "fev": 2, "fevereiro": 2, "mar": 3, "março": 3,
    "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8,
    "set": 9, "setembro": 9, "out": 10, "nov": 11, "dez": 12, "dezembro": 12,
}


def _normalize_date(raw: str) -> str:
    """Normalize any date format to DD/MM/YYYY. Empty string if unrecognized."""
    if not raw:
        return ""
    s = str(raw).strip()
    if not s:
        return ""

    # Already DD/MM/YYYY?
    m = re.match(r'^(\d{1,2})/(\d{1,2})/(\d{4})$', s)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"

    # DD-MM-YYYY, DD.MM.YYYY
    m = re.match(r'^(\d{1,2})[-.](\d{1,2})[-.](\d{4})$', s)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"

    # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
    m = re.match(r'^(\d{4})[-./](\d{1,2})[-./](\d{1,2})$', s)
    if m:
        y, mo, d = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"

    # DD-Mon-YYYY, DD Mon YYYY (Spanish/English/Portuguese month names or abbreviations)
    m = re.match(r'^(\d{1,2})[-\s]+([A-Za-zÀ-ÿ]+)[\.\-\s]+(\d{4})$', s)
    if m:
        d, mon, y = m.groups()
        mon_l = mon.lower().rstrip(".")
        if mon_l in _MONTHS:
            return f"{int(d):02d}/{_MONTHS[mon_l]:02d}/{y}"

    # Month DD, YYYY  (e.g. "April 23, 2026")
    m = re.match(r'^([A-Za-zÀ-ÿ]+)\s+(\d{1,2}),?\s+(\d{4})$', s)
    if m:
        mon, d, y = m.groups()
        mon_l = mon.lower()
        if mon_l in _MONTHS:
            return f"{int(d):02d}/{_MONTHS[mon_l]:02d}/{y}"

    # If nothing worked, return as-is (we tried)
    return s


def _normalize_number(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if "." in s and "," not in s:
        s = s.replace(".", ",")
    return s


def _extract_raw_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def _auto_correct_quantities(lines: list[dict]) -> list[dict]:
    """
    For each line, if quantity is 0, empty, or doesn't match linePrice/unitPrice,
    try to recompute it from the ratio. This fixes common LLM extraction errors
    where the model picks the wrong number from a messy layout.
    """
    for i, line in enumerate(lines, start=1):
        qty_raw = line.get("quantity", "").strip()
        unit_raw = line.get("unitPrice", "").strip()
        total_raw = line.get("linePrice", "").strip()

        # Need both unit and total to compute the ratio
        if not unit_raw or not total_raw:
            continue
        try:
            unit_num = float(unit_raw.replace(",", "."))
            total_num = float(total_raw.replace(",", "."))
        except ValueError:
            continue
        if unit_num <= 0:
            continue

        computed_qty = total_num / unit_num

        # Parse the quantity the LLM gave us
        qty_num = None
        if qty_raw:
            try:
                qty_num = float(qty_raw.replace(",", "."))
            except ValueError:
                qty_num = None

        needs_fix = False
        reason = ""
        if qty_num is None or qty_num <= 0:
            needs_fix = True
            reason = f"LLM gave invalid quantity ({qty_raw!r})"
        else:
            # Does it match the ratio?
            tolerance = max(computed_qty * 0.01, 0.01)
            if abs(qty_num - computed_qty) > tolerance:
                needs_fix = True
                reason = f"LLM quantity {qty_num} doesn't match ratio {computed_qty:.4f}"

        if needs_fix and computed_qty > 0:
            # Format the computed quantity nicely: integer if whole, else 2 decimals with comma
            if abs(computed_qty - round(computed_qty)) < 0.001:
                corrected = str(int(round(computed_qty)))
            else:
                corrected = f"{computed_qty:.2f}".replace(".", ",")
            print(f"[extractor] Auto-corrected Line {i} quantity: {qty_raw!r} → {corrected!r} ({reason})")
            line["quantity"] = corrected
            line["_qty_source"] = f"auto-corrected from linePrice/unitPrice"
    return lines


def _validate_lines(lines: list[dict]) -> list[str]:
    issues = []
    if not lines:
        issues.append("No order lines were extracted")
        return issues
    for i, line in enumerate(lines, start=1):
        if not line.get("hoffmannArticle", "").strip():
            issues.append(f"Line {i}: missing Hoffmann reference")

        qty_raw = line.get("quantity", "").strip()
        unit_raw = line.get("unitPrice", "").strip()
        total_raw = line.get("linePrice", "").strip()

        qty_num = unit_num = total_num = None

        # Parse quantity
        if not qty_raw:
            issues.append(f"Line {i}: missing quantity")
        else:
            try:
                qty_num = float(qty_raw.replace(",", "."))
                if qty_num <= 0:
                    issues.append(f"Line {i}: quantity must be greater than zero (got {qty_raw})")
                    qty_num = None
            except ValueError:
                issues.append(f"Line {i}: quantity is not numeric (got {qty_raw})")

        # Parse unit price
        if not unit_raw:
            issues.append(f"Line {i}: missing unit price")
        else:
            try:
                unit_num = float(unit_raw.replace(",", "."))
            except ValueError:
                issues.append(f"Line {i}: unit price is not numeric (got {unit_raw})")

        # Parse line price (optional but used for cross-check)
        if total_raw:
            try:
                total_num = float(total_raw.replace(",", "."))
            except ValueError:
                pass

        # Cross-check: linePrice / unitPrice should equal quantity (tolerance 1%)
        if qty_num is not None and unit_num is not None and total_num is not None and unit_num > 0:
            expected_qty = total_num / unit_num
            tolerance = max(expected_qty * 0.01, 0.01)
            if abs(qty_num - expected_qty) > tolerance:
                issues.append(
                    f"Line {i}: quantity mismatch — stated {qty_num}, but "
                    f"linePrice/unitPrice = {total_num}/{unit_num} = {expected_qty:.4f}"
                )
    return issues


def _build_error_response(
    reason: str,
    llm_data: dict | None,
    raw_text: str,
    client_info: dict,
    po_date: str,
) -> dict:
    llm_data = llm_data or {}
    return {
        "status": "error",
        "error": reason,
        "orderNumber": llm_data.get("orderNumber", ""),
        "deliveryDate": po_date,
        "customerName": llm_data.get("customerName", ""),
        "deliveryAddress": llm_data.get("deliveryAddress", ""),
        "shippingCustomerNumber": client_info.get("client_number", ""),
        "country": client_info.get("country", ""),
        "contact": {
            "name": llm_data.get("buyer", ""),
            "email": llm_data.get("email", ""),
            "phone": llm_data.get("phone", ""),
        },
        "partialExtraction": {
            "lines": llm_data.get("lines", []),
        },
        "rawTextPreview": raw_text[:1500],
    }


def _build_success_response(llm_data: dict, client_info: dict, po_date: str) -> str:
    head_fields = [
        "HEAD",
        client_info.get("client_number", ""),
        llm_data.get("orderNumber", ""),
        llm_data.get("buyer", ""),
        po_date,
        "",  # discount
        "",  # we-name
        "",  # shiptoaddress
        "",  # shiptoplace
        client_info.get("country", ""),
        "",  # shiptopostcode
        llm_data.get("email", ""),
    ]
    head = "|".join(head_fields)

    line_rows = []
    for line in llm_data.get("lines", []):
        row = "|".join([
            "LINE",
            str(line.get("hoffmannArticle", "")).strip(),
            _normalize_number(line.get("quantity", "")),
            "",  # customerArticle
            _normalize_number(line.get("unitPrice", "")),
            _normalize_number(line.get("linePrice", "")),
        ])
        line_rows.append(row)

    return head + "\r\n" + "\r\n".join(line_rows) + "\r\n"


def _fill_missing_refs_from_catalog(lines: list[dict], catalog: list[dict],
                                    min_confidence: int = 80) -> list[dict]:
    """For every line without a Hoffmann reference, try to resolve it using:
    1. The customer part number (normalized, space-insensitive lookup in catalog).
       Example: '724201125' → '724201 125' if that ref exists.
    2. Fuzzy match of the description against catalog product names (>= min_confidence).
    """
    if not catalog:
        return lines

    ref_index = build_ref_index(catalog)

    for line in lines:
        ref = str(line.get("hoffmannArticle", "")).strip()
        if ref:
            continue

        # Step 1: Try resolving from customerPartNumber
        cpn = str(line.get("customerPartNumber", "")).strip()
        if cpn:
            resolved = resolve_customer_part_number(cpn, ref_index)
            if resolved:
                line["hoffmannArticle"] = resolved
                line["_ref_source"] = f"customer-part-match ('{cpn}' → '{resolved}')"
                print(f"[extractor] Resolved customer part number: '{cpn}' → '{resolved}'")
                continue

        # Step 2: Fuzzy match by description
        desc = str(line.get("description", "")).strip()
        if not desc:
            continue
        found, score = find_reference_by_name(desc, catalog, min_confidence=min_confidence)
        if found:
            line["hoffmannArticle"] = found
            line["_ref_source"] = f"name-match ({score:.1f}%)"
            print(f"[extractor] Matched by name: '{desc[:60]}...' → {found} (score {score:.1f}%)")
        else:
            line["_ref_source"] = f"name-match failed (best {score:.1f}%)"
            print(f"[extractor] No name-match for '{desc[:60]}...' (best score {score:.1f}%)")

    return lines


# ─── Main entry point ────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str, catalog: list[dict], clients: list[dict]) -> tuple[bool, object]:
    """Returns (success, result). result is str on success, dict on error."""

    # ── 1. Extract raw text ─────────────────────────────────────────────
    try:
        raw_text = _extract_raw_text(pdf_path)
    except Exception as e:
        return False, {
            "status": "error",
            "error": f"Failed to read PDF: {type(e).__name__}: {e}",
            "orderNumber": "", "deliveryDate": "", "customerName": "",
            "deliveryAddress": "", "shippingCustomerNumber": "", "country": "",
            "contact": {"name": "", "email": "", "phone": ""},
            "partialExtraction": {"lines": []},
            "rawTextPreview": "",
        }

    # ── 2. Extract with OpenAI ──────────────────────────────────────────
    llm_data = None
    try:
        llm_data = extract_with_llm(raw_text)
        print(f"[extractor] LLM OK: orderNumber={llm_data.get('orderNumber')}, "
              f"poDate={llm_data.get('poDate')}, "
              f"lines={len(llm_data.get('lines', []))}")
    except Exception as e:
        print(f"[extractor] LLM call failed: {type(e).__name__}: {e}")
        client_info = find_client(raw_text[:2000], clients)
        return False, _build_error_response(
            f"OpenAI extraction failed: {type(e).__name__}: {e}",
            None, raw_text, client_info, "",
        )

    # ── 3. Normalize the PO date to DD/MM/YYYY ──────────────────────────
    po_date = _normalize_date(llm_data.get("poDate", ""))

    # ── 4. Client lookup ────────────────────────────────────────────────
    delivery_address = llm_data.get("deliveryAddress", "") or raw_text[:2000]
    client_info = find_client(delivery_address, clients)

    # ── 5. Fill missing references via catalog name match ───────────────
    lines = llm_data.get("lines", [])
    lines = _fill_missing_refs_from_catalog(lines, catalog, min_confidence=80)
    llm_data["lines"] = lines

    # ── 6. Auto-correct quantities using linePrice / unitPrice ratio ────
    lines = _auto_correct_quantities(lines)
    llm_data["lines"] = lines

    # ── 7. Validate ─────────────────────────────────────────────────────
    issues = []
    if not llm_data.get("orderNumber", "").strip():
        issues.append("Missing orderNumber")
    issues.extend(_validate_lines(lines))

    if issues:
        return False, _build_error_response(
            "; ".join(issues), llm_data, raw_text, client_info, po_date,
        )

    return True, _build_success_response(llm_data, client_info, po_date)
