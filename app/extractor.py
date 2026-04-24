"""
PDF extractor for Hoffmann purchase orders.

Strategy:
1. Extract text + tables with pdfplumber.
2. Try rule-based extraction first (Aludium, Astemo, generic).
3. Assess the quality of the rule-based result.
4. If quality is poor, fall back to OpenAI (gpt-4o-mini) for the whole PDF.
5. Match delivery address against clients table.
6. Assemble output in the required pipe-delimited format.
"""

from __future__ import annotations
import os
import re
import pdfplumber

from .catalog import validate_reference
from .clients import find_client


# ─── Regex patterns ───────────────────────────────────────────────────────────

ALUDIUM_LINE_RE = re.compile(
    r'^(\d+-\d+)\s+'
    r'(.+?)\s+'
    r'(\d{2}/\d{2}/\d{4})\s+'
    r'(\d+(?:[.,]\d+)?)\s+'
    r'(\w+)\s+'
    r'(\d+(?:[.,]\d+))\s+'
    r'(\d+(?:[.,]\d+))'
    r'(?:\n(.*))?$',
    re.DOTALL,
)

REF_AT_START_RE = re.compile(
    r'^(\d{5,})(?:\s+(\d+(?:/\d+)?|[A-Z]\d+(?:[A-Z]\d*)?|M\d+[A-Z0-9]*))?\s*-\s+'
)

REF_AT_END_RE = re.compile(
    r'(?:n[úu]mero\s+de\s+)?art[ií]?culo\s*:?\s*(\d{5,}(?:\s+[A-Z0-9/]+)?)',
    re.IGNORECASE,
)

ASTEMO_LINE_RE = re.compile(
    r'(\d{5,}(?:\s+\d+)?)\s*\n\s*(\d+[.,]\d+)\s+per\s+\d+',
    re.MULTILINE,
)

GENERIC_REF_RE = re.compile(
    r'\b(\d{5,})(?:\s+(\d+(?:/\d+)?|M\d+[A-Z0-9]*))?\b'
)

EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_date(raw: str) -> str:
    if not raw:
        return ""
    return raw.strip().replace("-", "/")


def _normalize_number(s: str) -> str:
    """Normalize number string: use comma as decimal separator."""
    if not s:
        return ""
    s = str(s).strip()
    if "." in s and "," not in s:
        s = s.replace(".", ",")
    return s


def _extract_reference(content: str, full_text: str) -> str:
    m = REF_AT_START_RE.match(content)
    if m:
        base, suffix = m.group(1), m.group(2)
        return (base + " " + suffix).strip() if suffix else base

    m = REF_AT_END_RE.search(full_text)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    m = GENERIC_REF_RE.search(full_text)
    if m:
        base, suffix = m.group(1), m.group(2)
        return (base + " " + suffix).strip() if suffix else base

    return ""


# ─── Format detection ─────────────────────────────────────────────────────────

def _detect_format(full_text: str) -> str:
    t = full_text.lower()
    if "no.pedido" in t or "pedido de compra" in t:
        return "aludium"
    if "purchase order number" in t:
        return "astemo"
    return "generic"


# ─── Header extraction (rule-based) ───────────────────────────────────────────

def _extract_header_aludium(full_text: str, tables: list) -> dict:
    h = {}

    m = re.search(r'No\.?Pedido\s+(\d+)', full_text, re.IGNORECASE)
    if m:
        h["orderNumber"] = m.group(1).strip()

    m = re.search(r'Fecha de orden\s+(\d{1,2}[/-]\d{2}[/-]\d{4})', full_text, re.IGNORECASE)
    if m:
        h["deliveryDate"] = _normalize_date(m.group(1))

    m = re.search(r'Comprador\s*\n\s*([^\n]+?)(?:\s+[A-Za-z0-9._%+-]+@|$)', full_text)
    if m:
        h["buyer"] = m.group(1).strip()
    else:
        for t in tables:
            for row in t:
                for cell in row:
                    if cell and str(cell).startswith("Comprador"):
                        parts = str(cell).split("\n", 1)
                        if len(parts) > 1:
                            h["buyer"] = parts[1].strip()
                            break

    for mm in EMAIL_RE.finditer(full_text):
        email = mm.group(0).lower()
        if not any(b in email for b in ("hofmann", "hoffmann", "normadat", "pagero", "info@aludium")):
            h["email"] = mm.group(0)
            break

    m = re.search(
        r'Direcci[oó]n\s+entrega\s*\n((?:[^\n]+\n){1,8})',
        full_text,
        re.IGNORECASE,
    )
    if m:
        lines = []
        for ln in m.group(1).split("\n"):
            if not ln.strip() or ln.strip().startswith("Proveedor"):
                break
            lines.append(ln.strip())
        h["deliveryAddress"] = " ".join(lines)

    if not h.get("deliveryAddress"):
        for t in tables:
            for row in t:
                for cell in row:
                    if cell and "Dirección entrega" in str(cell):
                        parts = str(cell).split("Dirección entrega", 1)[1]
                        h["deliveryAddress"] = " ".join(
                            ln.strip() for ln in parts.split("\n") if ln.strip()
                        )
                        break

    return h


def _extract_header_astemo(full_text: str, tables: list) -> dict:
    h = {}

    m = re.search(r'Purchase order number[:\s]+(\S+)', full_text, re.IGNORECASE)
    if m:
        h["orderNumber"] = m.group(1).strip()

    m = re.search(r'Creation date[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})', full_text, re.IGNORECASE)
    if m:
        h["deliveryDate"] = _normalize_date(m.group(1))

    m = re.search(r'Requester\s*:\s*(.+?)(?:\s+Sales\b|\s+Tel\b|\s+Email\b|\n)', full_text)
    if m:
        h["buyer"] = m.group(1).strip()

    for mm in EMAIL_RE.finditer(full_text):
        email = mm.group(0).lower()
        if ("hoffmann" not in email and "vendor" not in email and
            "proc-center" not in email and "payables" not in email):
            h["email"] = mm.group(0)
            break

    m = re.search(
        r'Delivery address\s*\n((?:[^\n]+\n){1,8})',
        full_text,
        re.IGNORECASE,
    )
    if m:
        lines = []
        for ln in m.group(1).split("\n"):
            s = ln.strip()
            if not s or s.startswith("Invoiced") or s.startswith("Terms"):
                break
            lines.append(s)
        h["deliveryAddress"] = " ".join(lines)

    return h


def _extract_header_generic(full_text: str, tables: list) -> dict:
    h = {}
    for pat in [
        r'No\.?Pedido\s+(\S+)',
        r'Purchase order number[:\s]+(\S+)',
        r'Order number[:\s]+(\S+)',
        r'N[º°]\s*pedido[:\s]+(\S+)',
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            h["orderNumber"] = m.group(1).strip()
            break

    for pat in [
        r'Fecha de orden\s+(\d{1,2}[/-]\d{2}[/-]\d{4})',
        r'Creation date[:\s]+(\d{2}[-/]\d{2}[-/]\d{4})',
        r'Date[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
    ]:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            h["deliveryDate"] = _normalize_date(m.group(1))
            break

    m = EMAIL_RE.search(full_text)
    if m:
        h["email"] = m.group(0)

    return h


# ─── Line extraction (rule-based) ─────────────────────────────────────────────

def _extract_lines_aludium(tables: list) -> list[dict]:
    lines = []
    for table in tables:
        for row in table:
            if not row:
                continue
            candidates = []
            for c in row:
                if c:
                    candidates.append(str(c).strip())
            joined = " ".join(str(c) for c in row if c).strip()
            if joined and joined not in candidates:
                candidates.append(joined)

            for text in candidates:
                m = ALUDIUM_LINE_RE.match(text)
                if m:
                    line_rel, content, date, qty, uom, price, amount, cont = m.groups()
                    full = content + (" " + cont if cont else "")
                    ref = _extract_reference(content, full)
                    lines.append({
                        "line_rel": line_rel,
                        "hoffmannArticle": ref,
                        "quantity": _normalize_number(qty),
                        "unitPrice": _normalize_number(price),
                        "linePrice": _normalize_number(amount),
                    })
                    break

    seen = set()
    out = []
    for l in lines:
        if l["line_rel"] in seen:
            continue
        seen.add(l["line_rel"])
        out.append(l)
    return out


def _extract_lines_astemo(full_text: str) -> list[dict]:
    lines = []
    for m in ASTEMO_LINE_RE.finditer(full_text):
        ref = re.sub(r'\s+', ' ', m.group(1)).strip()
        unit_price = _normalize_number(m.group(2))

        start = m.start()
        block_before = full_text[max(0, start - 400): start]

        qty_m = re.search(r'(\d+(?:[.,]\d+)?)\s+(?:Each|each|Un|UN)\b', block_before)
        quantity = _normalize_number(qty_m.group(1)) if qty_m else ""

        amt_m = None
        for am in re.finditer(r'(\d+[.,]\d+)\s+(?:EUR|eur)\b', block_before):
            amt_m = am
        line_price = _normalize_number(amt_m.group(1)) if amt_m else unit_price

        lines.append({
            "line_rel": f"{len(lines)+1}-1",
            "hoffmannArticle": ref,
            "quantity": quantity,
            "unitPrice": unit_price,
            "linePrice": line_price,
        })
    return lines


# ─── Quality assessment ───────────────────────────────────────────────────────

def _assess_quality(header: dict, lines: list[dict]) -> tuple[bool, str]:
    """
    Returns (is_good_enough, reason).
    Good means:
    - orderNumber present
    - at least one line extracted
    - every line has a reference, quantity and unit price
    - arithmetic qty * unit_price ≈ line_price (when all three present)
    """
    if not header.get("orderNumber"):
        return False, "no orderNumber"
    if not lines:
        return False, "no lines extracted"

    for i, line in enumerate(lines):
        if not line.get("hoffmannArticle"):
            return False, f"line {i+1} has no reference"
        if not line.get("quantity"):
            return False, f"line {i+1} has no quantity"
        if not line.get("unitPrice"):
            return False, f"line {i+1} has no unit price"

        try:
            qty = float(line["quantity"].replace(",", "."))
            unit = float(line["unitPrice"].replace(",", "."))
            if line.get("linePrice"):
                total = float(line["linePrice"].replace(",", "."))
                expected = qty * unit
                diff = abs(total - expected)
                if diff > 0.05 and (total == 0 or diff / max(total, 0.01) > 0.02):
                    return False, f"line {i+1} math mismatch ({qty} x {unit} != {total})"
        except (ValueError, AttributeError):
            pass

    return True, "ok"


# ─── Main entry point ────────────────────────────────────────────────────────

def extract_pdf(pdf_path: str, catalog: list[str], clients: list[dict]) -> str:
    """Extract a purchase order PDF and return the pipe-delimited string."""

    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        all_tables = []
        for page in pdf.pages:
            try:
                all_tables.extend(page.extract_tables() or [])
            except Exception:
                pass

    fmt = _detect_format(full_text)

    # ─── Rule-based extraction (ONLY for known formats) ──────────────────
    used_rules = False
    if fmt == "aludium":
        header = _extract_header_aludium(full_text, all_tables)
        lines = _extract_lines_aludium(all_tables)
        used_rules = True
    elif fmt == "astemo":
        header = _extract_header_astemo(full_text, all_tables)
        lines = _extract_lines_astemo(full_text)
        used_rules = True
    else:
        # For unknown formats: skip the unreliable generic rules
        # and go straight to the LLM below
        header = {}
        lines = []

    # ─── Quality check ───────────────────────────────────────────────────
    is_good, reason = _assess_quality(header, lines) if used_rules else (False, "unknown format")
    print(f"[extractor] format={fmt}, rules_used={used_rules}, quality_ok={is_good}, reason={reason}, lines={len(lines)}")

    # ─── LLM fallback (or main path for unknown formats) ─────────────────
    if not is_good:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[extractor] OPENAI_API_KEY not set — cannot fall back to LLM")
        else:
            print(f"[extractor] Calling OpenAI to extract this PDF...")
            try:
                from .llm import extract_with_llm
                llm_data = extract_with_llm(full_text)
                print(f"[extractor] LLM returned: orderNumber={llm_data.get('orderNumber')}, lines={len(llm_data.get('lines', []))}")

                # Fill missing header fields from LLM
                for field in ("orderNumber", "deliveryDate", "buyer", "email", "deliveryAddress"):
                    if not header.get(field) and llm_data.get(field):
                        header[field] = llm_data[field]

                llm_lines = [
                    {
                        "line_rel": f"{i+1}-1",
                        "hoffmannArticle": str(l.get("hoffmannArticle", "")).strip(),
                        "quantity": _normalize_number(str(l.get("quantity", ""))),
                        "unitPrice": _normalize_number(str(l.get("unitPrice", ""))),
                        "linePrice": _normalize_number(str(l.get("linePrice", ""))),
                    }
                    for i, l in enumerate(llm_data.get("lines", []))
                ]
                # Use LLM lines if rules failed or LLM found more
                if not lines or len(llm_lines) > len(lines):
                    lines = llm_lines
            except Exception as e:
                print(f"[extractor] LLM call failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

    # ─── Validate references against catalog ────────────────────────────
    for line in lines:
        if line.get("hoffmannArticle") and catalog:
            line["_in_catalog"] = validate_reference(line["hoffmannArticle"], catalog)

    # ─── Client lookup ──────────────────────────────────────────────────
    delivery_address = header.get("deliveryAddress", "")
    client_info = find_client(delivery_address, clients)

    # ─── Assemble output ────────────────────────────────────────────────
    head_fields = [
        "HEAD",
        client_info.get("client_number", ""),
        header.get("orderNumber", ""),
        header.get("buyer", ""),
        header.get("deliveryDate", ""),
        "",  # discount
        "",  # we-name
        "",  # shiptoaddress
        "",  # shiptoplace
        client_info.get("country", ""),
        "",  # shiptopostcode
        header.get("email", ""),
    ]
    head = "|".join(head_fields)

    line_rows = []
    for line in lines:
        row = "|".join([
            "LINE",
            line.get("hoffmannArticle", ""),
            line.get("quantity", ""),
            "",  # customerArticle
            line.get("unitPrice", ""),
            line.get("linePrice", ""),
        ])
        line_rows.append(row)

    return head + "\r\n" + "\r\n".join(line_rows)
