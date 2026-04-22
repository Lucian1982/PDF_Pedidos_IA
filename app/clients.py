import re
import pandas as pd
from rapidfuzz import fuzz, process


def load_clients(path: str) -> list[dict]:
    """
    Load the clients Excel.
    Expected columns: 'Direccion envio', 'Numero clinete', 'Pais'.
    Column names are matched case-insensitively and tolerant of spaces.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()

    # Flexible column detection
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "direcc" in cl and "envio" in cl:
            col_map["address"] = c
        elif "numero" in cl and ("clinete" in cl or "cliente" in cl):
            col_map["number"] = c
        elif "pais" in cl or "país" in cl:
            col_map["country"] = c

    clients = []
    for _, row in df.iterrows():
        addr = str(row.get(col_map.get("address", ""), "")).strip()
        num = str(row.get(col_map.get("number", ""), "")).strip()
        country = str(row.get(col_map.get("country", ""), "")).strip().upper()
        if addr and num:
            clients.append({
                "address": addr,
                "client_number": num,
                "country": country,
            })
    return clients


def extract_postal_codes(text: str) -> list[str]:
    """Extract all 4-5 digit numbers that look like postal codes."""
    return re.findall(r'\b(\d{4,5})\b', text or "")


def find_client(delivery_address: str, clients: list[dict]) -> dict:
    """
    Find the best matching client for a given delivery address.

    Strategy:
    1. Try postal code match first (fastest, most reliable).
    2. If multiple matches, refine with fuzzy address similarity.
    3. If no postal code match, fall back to pure fuzzy match.
    4. If nothing plausible, return empty values.
    """
    empty = {"client_number": "", "country": "", "address": ""}
    if not delivery_address or not clients:
        return empty

    postal_codes = extract_postal_codes(delivery_address)

    # Step 1: filter candidates by any matching postal code
    candidates = []
    for cp in postal_codes:
        candidates.extend([c for c in clients if cp in c["address"]])

    # Deduplicate by preserving order
    seen = set()
    unique = []
    for c in candidates:
        key = c["client_number"]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    candidates = unique

    # Step 2: if no candidates by postal code, use all clients (fuzzy fallback)
    if not candidates:
        candidates = clients

    # Step 3: single candidate wins
    if len(candidates) == 1:
        c = candidates[0]
        return {
            "client_number": c["client_number"],
            "country": c["country"],
            "address": c["address"],
        }

    # Step 4: fuzzy match among candidates (requires strong similarity)
    addresses = [c["address"] for c in candidates]
    result = process.extractOne(
        delivery_address,
        addresses,
        scorer=fuzz.token_set_ratio,
        score_cutoff=60,
    )
    if result:
        best_addr = result[0]
        best = next(c for c in candidates if c["address"] == best_addr)
        return {
            "client_number": best["client_number"],
            "country": best["country"],
            "address": best["address"],
        }

    return empty
