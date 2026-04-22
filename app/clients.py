import pandas as pd
import re
from rapidfuzz import fuzz, process


def load_clients(path: str) -> list[dict]:
    df = pd.read_excel(path, dtype=str)
    df.columns = df.columns.str.strip()
    clients = []
    for _, row in df.iterrows():
        clients.append({
            "address": str(row.get("Direccion envio", "")).strip(),
            "client_number": str(row.get("Numero clinete", "")).strip(),
            "country": str(row.get("Pais", "")).strip().upper(),
        })
    return clients


def extract_postal_code(text: str) -> str | None:
    match = re.search(r'\b(\d{4,5})\b', text)
    return match.group(1) if match else None


def find_client(delivery_address: str, clients: list[dict]) -> dict:
    if not delivery_address or not clients:
        return {"client_number": "", "country": "", "address": ""}

    pdf_cp = extract_postal_code(delivery_address)

    candidates = clients
    if pdf_cp:
        cp_matches = [c for c in clients if pdf_cp in c["address"]]
        if cp_matches:
            candidates = cp_matches

    if not candidates:
        return {"client_number": "", "country": "", "address": ""}

    if len(candidates) == 1:
        best = candidates[0]
    else:
        addresses = [c["address"] for c in candidates]
        result = process.extractOne(
            delivery_address,
            addresses,
            scorer=fuzz.token_set_ratio,
            score_cutoff=40
        )
        if result:
            best_address = result[0]
            best = next(c for c in candidates if c["address"] == best_address)
        else:
            best = candidates[0]

    return {
        "client_number": best["client_number"],
        "country": best["country"],
        "address": best["address"],
    }
