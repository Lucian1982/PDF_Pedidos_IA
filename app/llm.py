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
  "deliveryDate": "order/creation date in DD/MM/YYYY format",
  "buyer": "full name of the person who placed the order (not a department)",
  "email": "buyer's email. Skip supplier emails (hoffmann) and generic invoice/payables inboxes. If only a generic company inbox is available, use it.",
  "phone": "phone of the buyer or the requester",
  "customerName": "full legal name of the buyer company (e.g. 'Patentes Talgo, S.L.U.', 'TE Connectivity Componentes Electromecanicos LDA')",
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

CRITICAL RULES about hoffmannArticle:
- Hoffmann references have 5-6 digits, optionally followed by a space and a suffix (numeric, fraction, or M-style code).
  Examples: "759800", "759856 600", "640190 1/2", "082812 M12", "642229 8", "708205 300", "845020 18".
- The Hoffmann reference is embedded in the item description. It may appear:
  * At the beginning: "759800 - PALANCA DE UÑA"
  * At the end after a label: "Número de artículo: 845020 18", "Part number: 708205 300", "Your part number: 576705W"
  * Mixed with the brand: "HOFFMANN/HOLEX 708205 300", "HOLEX 759856 600"
- DO NOT use the customer's internal material code as hoffmannArticle (e.g. Talgo "10078071", TE "576705W" unless it clearly matches the Hoffmann pattern).
- If you cannot find a clear Hoffmann reference in a line, return "" (empty string) for that field.

FORMATTING RULES:
- Use comma (,) as decimal separator for ALL numbers (e.g. "12,65" not "12.65"). Convert dots to commas.
- Return EVERY line item, never skip one, even if some fields are empty.
- Do not include any text outside the JSON object.

PDF TEXT:
---
<<<PDF_TEXT>>>
---
"""


def extract_with_llm(full_text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Extract order data from raw PDF text using OpenAI.
    Returns a dict with all expected fields. Raises on OpenAI errors.
    """
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

    # Ensure all expected fields exist
    for f in ("orderNumber", "deliveryDate", "buyer", "email", "phone",
              "customerName", "deliveryAddress"):
        data.setdefault(f, "")
    data.setdefault("lines", [])

    for line in data["lines"]:
        line.setdefault("hoffmannArticle", "")
        line.setdefault("quantity", "")
        line.setdefault("unitPrice", "")
        line.setdefault("linePrice", "")

    return data
