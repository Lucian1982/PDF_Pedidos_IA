import re
import pandas as pd
from rapidfuzz import fuzz, process


def load_clients(path: str) -> list[dict]:
    """
    Load clients Excel.
    Expected columns: 'Direccion envio', 'Numero clinete', 'Pais'.
    """
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()

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
    return re.findall(r'\b(\d{4,5})\b', text or "")


def find_client(delivery_address: str, clients: list[dict]) -> dict:
    """
    Find the best-matching client for a given delivery address.
    Strategy: filter by postal code, then fuzzy match within the candidates.
    """
    empty = {"client_number": "", "country": "", "address": ""}
    if not delivery_address or not clients:
        return empty

    postal_codes = extract_postal_codes(delivery_address)

    candidates = []
    for cp in postal_codes:
        candidates.extend([c for c in clients if cp in c["address"]])

    seen = set()
    unique = []
    for c in candidates:
        if c["client_number"] not in seen:
            seen.add(c["client_number"])
            unique.append(c)
    candidates = unique

    if not candidates:
        candidates = clients

    if len(candidates) == 1:
        c = candidates[0]
        return {"client_number": c["client_number"], "country": c["country"], "address": c["address"]}

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
        return {"client_number": best["client_number"], "country": best["country"], "address": best["address"]}

    return empty
