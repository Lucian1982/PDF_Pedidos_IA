"""
OpenAI extractor for Hoffmann purchase orders.
Uses gpt-4o-mini with structured JSON output.
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
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT = (
    "You are a precise data-extraction engine for purchase orders (POs). "
    "The supplier is ALWAYS 'Hoffmann' (or Hoffmann Iberia / Hoffmann Group). "
    "The buyer varies: hundreds of different customers, each with their own layout and language "
    "(Spanish, English, Portuguese, Italian, French, etc.). "
    "Your job is to pull the header fields and the line items of the PO. "
    "Output valid JSON only. Never invent data. If a field is missing, return an empty string."
)

USER_PROMPT = """Extract the purchase order data from the following PDF text.

Return a JSON object with this exact shape:

{
  "orderNumber": "the PO number as it appears on the document",
  "deliveryDate": "the creation/order date in DD/MM/YYYY format",
  "buyer": "full name of the person who placed the order (NOT a department, NOT a generic title)",
  "email": "email of the buyer. If only a generic inbox appears, use that. Skip emails of the supplier (hoffmann) and emails of the invoicing/payables department.",
  "deliveryAddress": "full ship-to / delivery address as a single line, including street, postal code, city and country",
  "lines": [
    {
      "hoffmannArticle": "the Hoffmann article number",
      "quantity": "numeric quantity",
      "unitPrice": "unit price",
      "linePrice": "total amount for the line"
    }
  ]
}

CRITICAL RULES about the Hoffmann article number:
- Hoffmann references typically have 5-6 digits, sometimes followed by a numeric or alphanumeric suffix separated by a space.
  Examples: "759800", "759856 600", "640190 1/2", "082812 M12", "642229 8", "708205 300".
- The Hoffmann reference is usually embedded inside the item description. It may appear:
  * At the beginning: "759800 - PALANCA DE UÑA"
  * At the end after a label: "Número de artículo: 845020 18" or "Part number: 708205 300"
  * Mixed with a brand: "HOFFMANN/HOLEX 708205 300" or "HOLEX 759856 600"
  * As "Your part number" or "Supplier part number" field
- DO NOT use the customer's internal material code (e.g. Talgo "10078071", TE "576705W") as hoffmannArticle.
- If you cannot find a clear Hoffmann reference in the line, return an empty string for that field.

FORMATTING RULES:
- Use comma (,) as decimal separator for all numbers (e.g. "12,65" not "12.65").
- Return every line item, never skip one, even if some fields are empty.
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

    data.setdefault("orderNumber", "")
    data.setdefault("deliveryDate", "")
    data.setdefault("buyer", "")
    data.setdefault("email", "")
    data.setdefault("deliveryAddress", "")
    data.setdefault("lines", [])

    for line in data["lines"]:
        line.setdefault("hoffmannArticle", "")
        line.setdefault("quantity", "")
        line.setdefault("unitPrice", "")
        line.setdefault("linePrice", "")

    return data
