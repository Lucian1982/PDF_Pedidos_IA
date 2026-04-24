"""
Hoffmann PDF extractor.

Strategy (new, simplified):
1. Extract text from PDF with pdfplumber.
2. Send the text to OpenAI. Always.
3. If OpenAI fails → return error JSON with as much context as possible.
4. If OpenAI returns data but there are missing/empty required fields
   (orderNumber, at least one line with reference+quantity+price) → return error JSON.
5. If everything is good → return pipe-delimited HEAD|LINE|... output.
"""

from __future__ import annotations
import pdfplumber

from .llm import extract_with_llm
from .clients import find_client


def _normalize_number(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    # Dot to comma as decimal separator
    if "." in s and "," not in s:
        s = s.replace(".", ",")
    return s


def _extract_raw_text(pdf_path: str) -> str:
    """Extract all text from the PDF as a single string."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def _validate_lines(lines: list[dict]) -> list[str]:
    """
    Check each extracted line.
    Returns a list of issues. If empty → data is clean.
    """
    issues = []
    if not lines:
        issues.append("No order lines were extracted")
        return issues

    for i, line in enumerate(lines, start=1):
        if not line.get("hoffmannArticle", "").strip():
            issues.append(f"Line {i}: missing Hoffmann reference")
        if not line.get("quantity", "").strip():
            issues.append(f"Line {i}: missing quantity")
        if not line.get("unitPrice", "").strip():
            issues.append(f"Line {i}: missing unit price")

    return issues


def _build_error_response(
    reason: str,
    llm_data: dict | None,
    raw_text: str,
    client_info: dict,
) -> dict:
    """Build the error JSON response with maximum useful context."""
    llm_data = llm_data or {}
    return {
        "status": "error",
        "error": reason,
        "orderNumber": llm_data.get("orderNumber", ""),
        "deliveryDate": llm_data.get("deliveryDate", ""),
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


def _build_success_response(llm_data: dict, client_info: dict) -> str:
    """Build the pipe-delimited HEAD|LINE output for a successful extraction."""
    head_fields = [
        "HEAD",
        client_info.get("client_number", ""),
        llm_data.get("orderNumber", ""),
        llm_data.get("buyer", ""),
        llm_data.get("deliveryDate", ""),
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

    return head + "\r\n" + "\r\n".join(line_rows)


def extract_pdf(pdf_path: str, catalog: list[str], clients: list[dict]) -> tuple[bool, object]:
    """
    Main entry point.
    Returns a tuple (success: bool, result: str | dict).
    - If success=True, result is the pipe-delimited string.
    - If success=False, result is the error dict (to be returned as JSON).
    """
    # ── 1. Extract raw text ─────────────────────────────────────────────
    try:
        raw_text = _extract_raw_text(pdf_path)
    except Exception as e:
        return False, {
            "status": "error",
            "error": f"Failed to read PDF: {type(e).__name__}: {e}",
            "orderNumber": "",
            "deliveryDate": "",
            "customerName": "",
            "deliveryAddress": "",
            "shippingCustomerNumber": "",
            "country": "",
            "contact": {"name": "", "email": "", "phone": ""},
            "partialExtraction": {"lines": []},
            "rawTextPreview": "",
        }

    # ── 2. Extract with OpenAI (always) ─────────────────────────────────
    llm_data = None
    try:
        llm_data = extract_with_llm(raw_text)
        print(f"[extractor] LLM OK: orderNumber={llm_data.get('orderNumber')}, "
              f"lines={len(llm_data.get('lines', []))}, "
              f"customerName={llm_data.get('customerName')}")
    except Exception as e:
        print(f"[extractor] LLM call failed: {type(e).__name__}: {e}")
        # Try to do a client lookup on the raw text anyway
        client_info = find_client(raw_text[:2000], clients)
        return False, _build_error_response(
            f"OpenAI extraction failed: {type(e).__name__}: {e}",
            None,
            raw_text,
            client_info,
        )

    # ── 3. Client lookup (regardless of extraction result) ──────────────
    delivery_address = llm_data.get("deliveryAddress", "") or raw_text[:2000]
    client_info = find_client(delivery_address, clients)

    # ── 4. Validate: must have orderNumber + valid lines ────────────────
    issues = []
    if not llm_data.get("orderNumber", "").strip():
        issues.append("Missing orderNumber")
    issues.extend(_validate_lines(llm_data.get("lines", [])))

    if issues:
        return False, _build_error_response(
            "; ".join(issues),
            llm_data,
            raw_text,
            client_info,
        )

    # ── 5. All good: build success response ─────────────────────────────
    return True, _build_success_response(llm_data, client_info)
