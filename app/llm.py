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
  "customerName": "full legal name of the buyer company (e.g. 'Patentes Talgo, S.L.U.', 'TE Connectivity Componentes Electromecanicos LDA', 'EFAPEL - Empresa Fabril de Produtos Eléctricos, S.A.')",
  "customerVat": "the VAT/NIF/CIF/Tax ID of the CUSTOMER (the buyer company), never of Hoffmann. Look for labels like 'VAT', 'NIF', 'CIF', 'Tax ID', 'Tax number', 'C.I.F.', 'VAT/NIF'. Include the country prefix if present (e.g. 'PT501486429', 'ESB84528553', '500829136'). NEVER use Hoffmann's VAT numbers: 'B85500882', 'ESB85500882', 'B83727255', 'ESB83727255', 'PT980671566', '980671566'. If multiple VATs are present in the document, pick the one belonging to the buyer (the party receiving the goods), NOT Hoffmann.",
  "deliveryAddress": "full SHIP-TO / DELIVERY address where the goods must be delivered, as a single line, including street, postal code, city and country. This is the BUYER's warehouse/plant address, NOT the supplier's (Hoffmann) address and NOT the buyer's billing/invoice address. Look for labels like 'Dirección de entrega', 'Dirección de envío', 'Delivery address', 'Ship to', 'Endereço de entrega', 'Local de entrega', 'Local de descarga', 'Descarga', 'Adresse de livraison', 'Lieu de livraison', 'Lieferadresse'. If the document shows a 'Descarga:' block with just a city/postal code/location, use THAT as the delivery address even if it is shorter than the billing or supplier address. NEVER use the billing address of the customer (labels like 'Dirección de facturación', 'Invoice address', 'Endereço da Fatura', 'Adresse de facturation') and NEVER use the address of HOFFMANN (the supplier).",
  "lines": [
    {
      "hoffmannArticle": "the Hoffmann article number (empty string if not present)",
      "customerPartNumber": "the customer's own part/reference number for this item, as shown on the document. Look for labels like 'Your part number', 'Your reference', 'Customer part number', 'Your PN', 'Nº material', 'Material', 'Código'. Return the value exactly as it appears, including any trailing letters or special characters. Empty string if not present.",
      "description": "the FULL item description as it appears in the PDF (e.g. 'PALANCA DE UÑA', 'Cepillo fino Alambre de latón ondulado', 'BOQUILLA AIRMIX PORTAINSERTO INOX'). Include brand and size/variant when present.",
      "quantity": "numeric quantity - the number of units ordered. NEVER zero.",
      "unitPrice": "unit price",
      "linePrice": "total amount for the line"
    }
  ]
}

CRITICAL RULES about hoffmannArticle:
- Hoffmann references have 5-6 digits, optionally followed by a space and a suffix (numeric, fraction, or M-style code).
  Examples: "759800", "759856 600", "640190 1/2", "082812 M12", "642229 8", "708205 300", "845020 18", "626089 9".
- The Hoffmann reference is embedded in the item description. It may appear:
  * At the beginning: "759800 - PALANCA DE UÑA"
  * At the end of the description: "JOGO DE CHAVES UMBRAKO 626089 9"
  * After a label: "Número de artículo: 845020 18", "Part number: 708205 300", "Nº peça de fabricante: 626089 9", "Manufacturer part number: 759800", "MFG P/N: 626089 9"
  * Mixed with the brand: "HOFFMANN/HOLEX 708205 300", "HOLEX 759856 600"

- ⚠️ DO NOT confuse the customer's INTERNAL material code with the Hoffmann reference. These are NOT Hoffmann references:
  * Codes labelled "Your part number", "Your reference", "Customer part number", "Your PN"
  * Codes in columns labelled "Material", "Item number", "Material number", "Código", "Código interno"
  * Codes with 7+ digits and no space-suffix (e.g. "00080218600" is the customer's material code, NOT Hoffmann)
  * Examples of customer-internal codes (DO NOT use as hoffmannArticle): "10078071" (Talgo), "576705W" / "576705 W" (TE), "00080218600" (Continental), "6074", "5388031" (supplier numbers)

- Look for the Hoffmann reference SPECIFICALLY in:
  * The description text itself (read each word)
  * Fields labelled "Nº peça de fabricante" / "Manufacturer part number" / "Hersteller-Teilenummer"
  * Fields labelled with "HOFFMANN" or "HOLEX" near a code

- If the line shows ONLY a customer-internal code and the description does NOT contain a clear Hoffmann reference, return "" (empty string) for hoffmannArticle. The system will look it up later via the description.

CRITICAL RULES about quantity:
- The quantity is the NUMBER OF UNITS ORDERED. It is ALWAYS greater than zero.
- If the line shows two numbers like "0 5 PC" or "0 5 UN", the quantity is 5 (the non-zero one). The 0 is a column artifact, position number, or placeholder — NEVER the quantity.
- If the line shows "5 PC", "3 UN", "2,00 Each", "1.00 UN", the number right before the unit (PC, UN, Each, pcs) is the quantity.
- If you cannot confidently determine a positive quantity, return "" (empty string), never "0".

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
              "customerName", "customerVat", "deliveryAddress"):
        data.setdefault(f, "")
    data.setdefault("lines", [])

    for line in data["lines"]:
        line.setdefault("hoffmannArticle", "")
        line.setdefault("customerPartNumber", "")
        line.setdefault("description", "")
        line.setdefault("quantity", "")
        line.setdefault("unitPrice", "")
        line.setdefault("linePrice", "")

    return data
