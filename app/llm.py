"""
OpenAI-based extraction for Hoffmann purchase orders.
Uses gpt-4o-mini with structured JSON output. Always, for all PDFs.
"""

from __future__ import annotations
import os
import json
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT = (
    "You are a precise data-extraction engine for purchase orders (POs). "
    "The supplier is ALWAYS 'Hoffmann' (or Hoffmann Iberia / Hoffmann Group). "
    "The buyer is a customer company; every customer has its own layout and language "
    "(Spanish, English, Portuguese, Italian, French, German, etc.). "
    "Your job is to extract the header fields, company/contact info, and line items. "
    "Output valid JSON only. Never invent data. If a field is missing, return an empty string."
)

USER_PROMPT = """Extract the purchase order data from the following PDF text.

Return a JSON object with this EXACT shape:

{
  "orderNumber": "PO number as shown on the document",
  "poDate": "the DATE WHEN THE PO WAS CREATED/ISSUED (not delivery date, not 'date received', not item-level dates). Typical labels: 'Fecha de orden', 'Data da Ordem', 'Creation date', 'Document date', 'Order date', 'Fecha del pedido', 'Número del plan de entregas/Fecha'. Return it in DD/MM/YYYY format. Convert any format ('23-04-2026', '23-Abr-2026', '2026-04-23', 'April 23, 2026', '16.07.2024') into DD/MM/YYYY.",
  "buyer": "full name of the person who placed the order (not a department)",
  "email": "buyer's email. Skip supplier emails (hoffmann) and generic invoice/payables inboxes. If only a generic company inbox is available, use it.",
  "phone": "phone of the buyer or the requester",
  "customerName": "full legal name of the buyer company (e.g. 'Patentes Talgo, S.L.U.', 'TE Connectivity Componentes Electromecanicos LDA')",
  "deliveryAddress": "full ship-to / delivery address as a single line, including street, postal code, city and country",
  "lines": [
    {
      "hoffmannArticle": "the Hoffmann article number (empty string if not present)",
      "description": "the FULL item description as it appears in the PDF (e.g. 'PALANCA DE UÑA', 'Cepillo fino Alambre de latón ondulado', 'BOQUILLA AIRMIX PORTAINSERTO INOX'). Include brand and size/variant when present.",
      "quantity": "numeric quantity",
      "unitPrice": "unit price",
      "linePrice": "total amount for the line"
    }
  ]
}

CRITICAL RULES about hoffmannArticle:
- Hoffmann references have 5-6 digits, optionally followed by a space and a suffix (numeric, fraction, or M-style code).
  Examples: "759800", "759856 600", "640190 1/2", "082812 M12", "642229 8", "708205 300", "845020 18".
- The Hoffmann reference is embedded in the item description. It may appear:
  * At the beginning: "759800 - PALANCA DE UÑA"
  * At the end after a label: "Número de artículo: 845020 18", "Part number: 708205 300"
  * Mixed with the brand: "HOFFMANN/HOLEX 708205 300", "HOLEX 759856 600"
- DO NOT use the customer's internal material code as hoffmannArticle (e.g. Talgo "10078071", TE "576705W" unless it clearly matches the Hoffmann pattern).
- If you cannot find a clear Hoffmann reference in a line, return "" (empty string). The description will be used later to look it up.

FORMATTING RULES:
- Use comma (,) as decimal separator for ALL numbers (e.g. "12,65" not "12.65"). Convert dots to commas.
- poDate MUST be in DD/MM/YYYY format (e.g. "23/04/2026", "16/07/2024"). Convert any input format to this.
- Return EVERY line item, never skip one, even if some fields are empty.
- Do not include any text outside the JSON object.

PDF TEXT:
---
<<<PDF_TEXT>>>
---
"""


def extract_with_llm(full_text: str, model: str = "gpt-4o-mini") -> dict:
    client = _get_client()
    text = full_text[:15000]
    user = USER_PROMPT.replace("<<<PDF_TEXT>>>", text)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    data = json.loads(raw)

    # Default every expected field
    for f in ("orderNumber", "poDate", "buyer", "email", "phone",
              "customerName", "deliveryAddress"):
        data.setdefault(f, "")
    data.setdefault("lines", [])

    for line in data["lines"]:
        line.setdefault("hoffmannArticle", "")
        line.setdefault("description", "")
        line.setdefault("quantity", "")
        line.setdefault("unitPrice", "")
        line.setdefault("linePrice", "")

    return data
