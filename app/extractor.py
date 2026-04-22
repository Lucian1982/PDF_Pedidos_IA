import re
import os
import json
import pdfplumber
from openai import OpenAI

from .catalog import find_reference
from .clients import find_client

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ─── Header extraction ────────────────────────────────────────────────────────

def extract_header_with_rules(full_text: str) -> dict:
    """Try to extract header fields using regex patterns."""
    header = {}

    # Order number: look for 'No.Pedido', 'Purchase order number', etc.
    for pattern in [
        r'No\.?Pedido\s+(\d+)',
        r'Purchase order number[:\s]+(\d+)',
        r'Pedido\s+(\d{5,})',
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            header["orderNumber"] = m.group(1).strip()
            break

    # Date: fecha de orden / creation date
    for pattern in [
        r'Fecha de orden\s+(\d{1,2}/\d{2}/\d{4})',
        r'Creation date[:\s]+(\d{2}-\d{2}-\d{4})',
        r'Fecha de orden\s+(\d{2}-\d{2}-\d{4})',
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            date_raw = m.group(1).strip()
            # Normalize to DD/MM/YYYY
            header["deliveryDate"] = date_raw.replace("-", "/")
            break

    # Buyer name
    for pattern in [
        r'Comprador\s*\n\s*([^\n]+)',
        r'Requester[:\s]+([A-ZÁÉÍÓÚÑ][^\n]+)',
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            header["buyer"] = m.group(1).strip()
            break

    # Email
    m = re.search(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', full_text)
    if m:
        header["email"] = m.group(1).strip()

    return header


def extract_header_with_llm(full_text: str) -> dict:
    """Use OpenAI to extract header fields when regex fails."""
    prompt = f"""Extract the following fields from this purchase order text and return ONLY a JSON object with no markdown or explanation:
- orderNumber: the purchase order number (string)
- deliveryDate: the order creation date in DD/MM/YYYY format (string)
- buyer: the name of the person who made the order (string)
- email: the email address of the buyer (string)
- deliveryAddress: the full delivery address as a single string (string)

Purchase order text:
{full_text[:3000]}

Return ONLY valid JSON like: {{"orderNumber":"...","deliveryDate":"...","buyer":"...","email":"...","deliveryAddress":"..."}}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def extract_delivery_address(full_text: str) -> str:
    """Extract the delivery address block from the PDF text."""
    # Look for delivery address section
    for pattern in [
        r'Direcci[oó]n entrega\s*\n((?:[^\n]+\n){2,6})',
        r'Delivery address\s*\n((?:[^\n]+\n){2,6})',
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


# ─── Line extraction ──────────────────────────────────────────────────────────

def parse_number(s: str) -> str:
    """Normalize number string: strip spaces, keep comma as decimal separator."""
    return s.strip() if s else ""


def extract_lines_from_table(tables: list, catalog: list[str]) -> list[dict]:
    """
    Extract order lines from pdfplumber tables.
    Returns list of dicts with hoffmannArticle, quantity, unitPrice, linePrice.
    """
    lines = []
    line_pattern = re.compile(r'^\d+-\d+$')  # matches "1-1", "2-1", etc.

    for table in tables:
        if not table:
            continue
        for row in table:
            if not row:
                continue
            # Clean cells
            cells = [str(c).strip() if c else "" for c in row]

            # Skip header rows
            if any(h in cells[0].lower() for h in ["line", "artículo", "artculo", "item", "order"]):
                continue

            # Check if first cell looks like a line number "1-1", "2-1"
            is_line_row = line_pattern.match(cells[0]) if cells[0] else False

            # For Aludium format: Line-Rel | Description | Date | Qty | UdM | UnitPrice | Amount
            if is_line_row and len(cells) >= 6:
                description = cells[1] if len(cells) > 1 else ""
                quantity = cells[3] if len(cells) > 3 else ""
                unit_price = cells[5] if len(cells) > 5 else ""
                line_price = cells[6] if len(cells) > 6 else ""

                ref = find_reference(description, catalog)
                if ref or description:
                    lines.append({
                        "hoffmannArticle": ref or "",
                        "description_raw": description,
                        "quantity": parse_number(quantity),
                        "unitPrice": parse_number(unit_price),
                        "linePrice": parse_number(line_price),
                        "needs_llm": ref is None,
                    })

            # For Astemo format: Order line # | Item number | Qty | Unit | Description | Price | ...
            elif cells[0] and re.match(r'^\d+$', cells[0]) and len(cells) >= 5:
                # Item number may be in column 1
                item_number = cells[1] if len(cells) > 1 else ""
                description = cells[4] if len(cells) > 4 else ""
                quantity = cells[2] if len(cells) > 2 else ""
                unit_price = cells[5] if len(cells) > 5 else ""
                line_price = cells[6] if len(cells) > 6 else ""

                # Try catalog lookup on item number first, then description
                ref = find_reference(item_number + " " + description, catalog)
                if ref or description:
                    lines.append({
                        "hoffmannArticle": ref or "",
                        "description_raw": description,
                        "quantity": parse_number(quantity),
                        "unitPrice": parse_number(unit_price),
                        "linePrice": parse_number(line_price),
                        "needs_llm": ref is None,
                    })

    return lines


def extract_lines_with_llm(full_text: str, catalog: list[str]) -> list[dict]:
    """Use OpenAI to extract lines when table parsing fails or misses references."""
    prompt = f"""Extract all order lines from this purchase order text. For each line find:
- hoffmannArticle: the Hoffmann article/item reference number (digits, may include space and suffix like "845020 18" or "708205 300"). Look for patterns like "Número de artículo:", "Item number:", or numbers embedded in descriptions.
- quantity: the quantity ordered (number)
- unitPrice: the unit price (number with comma decimal)
- linePrice: the total line price (number with comma decimal)

Return ONLY a JSON array, no markdown. Example:
[{{"hoffmannArticle":"845020 18","quantity":"15","unitPrice":"6,32","linePrice":"94,80"}}]

Purchase order text:
{full_text[:4000]}"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
    llm_lines = json.loads(raw)

    # Validate references against catalog
    result = []
    for line in llm_lines:
        ref_raw = line.get("hoffmannArticle", "")
        # Confirm ref exists in catalog, else try lookup
        confirmed = find_reference(ref_raw, catalog) or ref_raw
        result.append({
            "hoffmannArticle": confirmed,
            "quantity": line.get("quantity", ""),
            "unitPrice": line.get("unitPrice", ""),
            "linePrice": line.get("linePrice", ""),
            "needs_llm": False,
        })
    return result


# ─── Main extraction function ─────────────────────────────────────────────────

def extract_pdf(pdf_path: str, catalog: list[str], clients: list[dict]) -> str:
    """
    Main function: extracts a purchase order PDF and returns the pipe-delimited output.
    """
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        all_tables = []
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)

    # ── 1. Extract header ──────────────────────────────────────────────────
    header = extract_header_with_rules(full_text)

    # Fill missing header fields with LLM
    needed = {"orderNumber", "deliveryDate", "buyer", "email"}
    missing = needed - set(k for k, v in header.items() if v)
    if missing:
        try:
            llm_header = extract_header_with_llm(full_text)
            for field in missing:
                if llm_header.get(field):
                    header[field] = llm_header[field]
            # Always get delivery address from LLM header if available
            if "deliveryAddress" not in header and llm_header.get("deliveryAddress"):
                header["deliveryAddress"] = llm_header["deliveryAddress"]
        except Exception as e:
            print(f"LLM header extraction failed: {e}")

    # ── 2. Extract delivery address and match client ───────────────────────
    delivery_address = header.get("deliveryAddress") or extract_delivery_address(full_text)
    client_info = find_client(delivery_address, clients)

    # ── 3. Extract order lines ─────────────────────────────────────────────
    lines = extract_lines_from_table(all_tables, catalog)

    # If no lines found from tables, use LLM
    if not lines:
        try:
            lines = extract_lines_with_llm(full_text, catalog)
        except Exception as e:
            print(f"LLM line extraction failed: {e}")

    # For lines that need LLM (reference not found in catalog), use LLM fallback
    unresolved = [l for l in lines if l.get("needs_llm")]
    if unresolved:
        try:
            # Build text snippet for just the unresolved descriptions
            snippet = "\n".join(l["description_raw"] for l in unresolved)
            llm_refs = extract_lines_with_llm(snippet, catalog)
            # Map back by position
            for i, line in enumerate(unresolved):
                if i < len(llm_refs):
                    line["hoffmannArticle"] = llm_refs[i].get("hoffmannArticle", "")
                line["needs_llm"] = False
        except Exception as e:
            print(f"LLM fallback for unresolved lines failed: {e}")

    # ── 4. Build output ────────────────────────────────────────────────────
    head = "|".join([
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
    ])

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
