"""
OpenAI fallback for PDFs where rule-based extraction is not good enough.
Uses gpt-4o-mini with a structured JSON schema so the output is deterministic.
"""

from __future__ import annotations
import os
import json
import re
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


PROMPT = """You extract purchase order data from a PDF's text. The PDFs are always for \
a supplier called Hoffmann, but the buyer (customer) varies: each customer has its own \
layout and language (Spanish, English, Portuguese, etc.).

Return ONLY a JSON object, no markdown, no comments. Use this exact schema:

{
  "orderNumber": "<purchase order number>",
  "deliveryDate": "<creation/order date as DD/MM/YYYY>",
  "buyer": "<full name of the person who placed the order>",
  "email": "<email of the buyer, not a generic inbox>",
  "deliveryAddress": "<full ship-to / delivery address as a single line>",
  "lines": [
    {
      "hoffmannArticle": "<Hoffmann article number, digits, may include a space and a \
suffix like '642229 8' or 'M12' or '1/2'>",
      "quantity": "<numeric quantity, use comma as decimal separator if decimal>",
      "unitPrice": "<unit price, comma as decimal separator>",
      "linePrice": "<total for the line, comma as decimal separator>"
    }
  ]
}

Rules:
- Hoffmann article numbers typically have 5-6 digits, sometimes followed by a \
numeric or alphanumeric suffix (e.g. "759856 600", "640190 1/2", "082812 M12").
- The Hoffmann article is usually embedded in the item description. It may appear at \
the start (e.g. "759800 - PALANCA DE UÑA") or at the end (e.g. \
"Número de artículo: 845020 18" or "HOFFMANN/HOLEX 708205 300").
- If a customer reference is shown (e.g. an internal SAP code), DO NOT use it as \
hoffmannArticle. Use ONLY the Hoffmann reference.
- Use comma (,) as decimal separator for all numbers.
- If a field is not present, use an empty string "".
- Return every line item found, never skip any.

PDF text:
---
{text}
---

JSON:"""


def extract_with_llm(full_text: str, model: str = "gpt-4o-mini") -> dict:
    """
    Extract order data using OpenAI.
    Returns a dict with orderNumber, deliveryDate, buyer, email, deliveryAddress, lines.
    """
    client = _get_client()
    # Truncate very long texts to stay within reasonable token limits
    text = full_text[:15000]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise data extraction engine. Output only valid JSON."},
            {"role": "user", "content": PROMPT.format(text=text)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    # Remove possible markdown fences
    raw = re.sub(r'^```(?:json)?|```$', '', raw, flags=re.MULTILINE).strip()
    data = json.loads(raw)

    # Normalize: ensure all expected keys exist
    data.setdefault("orderNumber", "")
    data.setdefault("deliveryDate", "")
    data.setdefault("buyer", "")
    data.setdefault("email", "")
    data.setdefault("deliveryAddress", "")
    data.setdefault("lines", [])

    # Normalize line keys
    for line in data["lines"]:
        line.setdefault("hoffmannArticle", "")
        line.setdefault("quantity", "")
        line.setdefault("unitPrice", "")
        line.setdefault("linePrice", "")

    return data
