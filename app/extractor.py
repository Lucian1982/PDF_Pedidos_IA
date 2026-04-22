"""
PDF extractor for Hoffmann purchase orders.

Strategy (in order of confidence):
1. Extract text + tables with pdfplumber.
2. Identify document format by keywords (Aludium-style, Astemo-style, generic).
3. Parse header fields with format-specific regex, fall back to generic regex.
4. Parse line items with format-specific regex.
5. Validate references against the catalog (when available).
6. Match delivery address against clients table.
7. Assemble output in the required pipe-delimited format.

No LLM calls are required; the rules handle the known formats. A hook is
provided (extractor._llm_rescue) for future OpenAI fallback.
"""

from __future__ import annotations
import os
import re
import json
import pdfplumber

from .catalog import validate_reference
from .clients import find_client


# ─── Regex patterns ───────────────────────────────────────────────────────────

# Aludium line: "1-1 759800 - PALANCA 20/04/2026 8 PCS 13,77 110,16[\n continuation]"
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

# Reference at start: "759800 - ..." or "759856 600 - ..." or "640190 1/2 - ..."
REF_AT_START_RE = re.compile(
    r'^(\d{5,})(?:\s+(\d+(?:/\d+)?|[A-Z]\d+(?:[A-Z]\d*)?|M\d+[A-Z0-9]*))?\s*-\s+'
)

# Reference at end: "... Número de artículo: 845020 18."
REF_AT_END_RE = re.compile(
    r'(?:n[úu]mero\s+de\s+)?art[ií]?culo\s*:?\s*(\d{5,}(?:\s+[A-Z0-9/]+)?)',
    re.IGNORECASE,
)

# Astemo-style reference in text block: "708205 300\n24.87 per 1"
ASTEMO_LINE_RE = re.compile(
    r'(\d{5,}(?:\s+\d+)?)\s*\n\s*(\d+[.,]\d+)\s+per\s+\d+',
    re.MULTILINE,
)

# Generic fallback: any 5+ digit number with optional numeric or M-style suffix
GENERIC_REF_RE = re.compile(
    r'\b(\d{5,})(?:\s+(\d+(?:/\d+)?|M\d+[A-Z0-9]*))?\b'
)

EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_date(raw: str) -> str:
    """Normalize date to DD/MM/YYYY."""
    if not raw:
        return ""
    return raw.strip().replace("-", "/")


def _extract_reference(content: str, full_text: str) -> str:
    """Try several strategies to find the Hoffmann reference in a line."""
    # 1) Reference at the start of the line content
    m = REF_AT_START_RE.match(content)
    if m:
        base, suffix = m.group(1), m.group(2)
        return (base + " " + suffix).strip() if suffix else base

    # 2) Reference explicitly marked with "artículo:" / "article:"
    m = REF_AT_END_RE.search(full_text)
    if m:
        return re.sub(r'\s+', ' ', m.group(1)).strip()

    # 3) Any 5+ digit number with optional suffix (last resort)
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


# ─── Header extraction ────────────────────────────────────────────────────────

def _extract_header_aludium(full_text: str, tables: list) -> dict:
    h = {}

    m = re.search(r'No\.?Pedido\s+(\d+)', full_text, re.IGNORECASE)
    if m:
        h["orderNumber"] = m.group(1).strip()

    m = re.search(r'Fecha de orden\s+(\d{1,2}[/-]\d{2}[/-]\d{4})', full_text, re.IGNORECASE)
    if m:
        h["deliveryDate"] = _normalize_date(m.group(1))

    # Buyer: "Comprador\n<name>" OR from tables: cell "Comprador\n<name>"
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

    m = EMAIL_RE.search(full_text)
    # Take the first email that is NOT the Hoffmann contact
    for mm in EMAIL_RE.finditer(full_text):
        email = mm.group(0)
        if "hofmann" not in email.lower() and "hoffmann" not in email.lower() and "normadat" not in email.lower() and "pagero" not in email.lower() and "info@aludium" not in email.lower():
            h["email"] = email
            break

    # Delivery address: section "Dirección entrega"
    m = re.search(
        r'Direcci[oó]n\s+entrega\s*\n((?:[^\n]+\n){1,8})',
        full_text,
        re.IGNORECASE,
    )
    if m:
        block = m.group(1)
        # Stop at "Proveedor" or empty line
        lines = []
        for ln in block.split("\n"):
            if not ln.strip():
                break
            if ln.strip().startswith("Proveedor"):
                break
            lines.append(ln.strip())
        h["deliveryAddress"] = " ".join(lines)
    # Alternative: look inside tables
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

    # Buyer: "Requester: DIAS JOÃO" on its own. Split by whitespace until we hit
    # a typical Astemo delimiter word.
    m = re.search(r'Requester\s*:\s*(.+?)(?:\s+Sales\b|\s+Tel\b|\s+Email\b|\n)', full_text)
    if m:
        h["buyer"] = m.group(1).strip()

    for mm in EMAIL_RE.finditer(full_text):
        email = mm.group(0)
        if "hoffmann" not in email.lower() and "astemo.com" in email.lower() and "vendor" not in email.lower() and "proc-center" not in email.lower() and "payables" not in email.lower():
            h["email"] = email
            break

    # Delivery address
    m = re.search(
        r'Delivery address\s*\n((?:[^\n]+\n){1,8})',
        full_text,
        re.IGNORECASE,
    )
    if m:
        lines = []
        for ln in m.group(1).split("\n"):
            if not ln.strip() or ln.strip().startswith("Invoiced") or ln.strip().startswith("Terms"):
                break
            lines.append(ln.strip())
        # Remove columns that may be glued: "Astemo... Astemo... Pdf format to"
        # Just take the first address block
        h["deliveryAddress"] = " ".join(lines)

    return h


def _extract_header_generic(full_text: str, tables: list) -> dict:
    h = {}
    # Try all known patterns
    patterns_order = [
        r'No\.?Pedido\s+(\S+)',
        r'Purchase order number[:\s]+(\S+)',
        r'Order number[:\s]+(\S+)',
        r'N[º°]\s*pedido[:\s]+(\S+)',
    ]
    for pat in patterns_order:
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


# ─── Line extraction ──────────────────────────────────────────────────────────

def _extract_lines_aludium(tables: list) -> list[dict]:
    """Parse Aludium-style tables where each line is a single text blob."""
    lines = []
    for table in tables:
        for row in table:
            if not row:
                continue
            # Try each cell, because pdfplumber sometimes keeps the whole row
            # in a single cell, sometimes splits it.
            candidates = []
            for c in row:
                if c:
                    candidates.append(str(c).strip())
            # Also try the full row joined
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
                        "quantity": qty.strip(),
                        "unitPrice": price.strip(),
                        "linePrice": amount.strip(),
                    })
                    break  # don't parse the same line twice
    # Deduplicate by line_rel
    seen = set()
    out = []
    for l in lines:
        if l["line_rel"] in seen:
            continue
        seen.add(l["line_rel"])
        out.append(l)
    return out


def _extract_lines_astemo(full_text: str) -> list[dict]:
    """Parse Astemo-style: lines are text blocks with 'ref\\nprice per N'."""
    lines = []

    # Find all "NNNNNN [suffix]\n<price> per <N>" patterns
    for m in ASTEMO_LINE_RE.finditer(full_text):
        ref = re.sub(r'\s+', ' ', m.group(1)).strip()
        unit_price = m.group(2).strip().replace(".", ",")

        # Look backward from the match for quantity and total
        start = m.start()
        block_before = full_text[max(0, start - 400): start]

        # Quantity: "X.XX Each" or "X Each"
        qty_m = re.search(r'(\d+(?:[.,]\d+)?)\s+(?:Each|each|Un|UN)\b', block_before)
        quantity = qty_m.group(1).replace(".", ",") if qty_m else ""

        # Total amount: last number followed by EUR before the ref
        amt_m = None
        for am in re.finditer(r'(\d+[.,]\d+)\s+(?:EUR|eur)\b', block_before):
            amt_m = am
        line_price = amt_m.group(1).replace(".", ",") if amt_m else unit_price

        # Skip total lines (whole-order total) — they'll come after "Total"
        # The ASTEMO_LINE_RE should not match the order total because
        # total has no "per" suffix. OK.

        lines.append({
            "line_rel": f"{len(lines)+1}-1",
            "hoffmannArticle": ref,
            "quantity": quantity,
            "unitPrice": unit_price,
            "linePrice": line_price,
        })
    return lines


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

    # Header
    if fmt == "aludium":
        header = _extract_header_aludium(full_text, all_tables)
    elif fmt == "astemo":
        header = _extract_header_astemo(full_text, all_tables)
    else:
        header = _extract_header_generic(full_text, all_tables)

    # Lines
    if fmt == "aludium":
        lines = _extract_lines_aludium(all_tables)
    elif fmt == "astemo":
        lines = _extract_lines_astemo(full_text)
    else:
        # Try both
        lines = _extract_lines_aludium(all_tables)
        if not lines:
            lines = _extract_lines_astemo(full_text)

    # Validate references (optional: mark if not in catalog)
    for line in lines:
        if line["hoffmannArticle"] and catalog:
            # Don't block if not found; just record
            line["_in_catalog"] = validate_reference(line["hoffmannArticle"], catalog)

    # Client lookup
    delivery_address = header.get("deliveryAddress", "")
    client_info = find_client(delivery_address, clients)

    # ── Assemble output ─────────────────────────────────────────────────
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
