import pandas as pd


def load_catalog(path: str) -> list[str]:
    """
    Loads the Hoffmann catalog.
    First column (Artikelnummer) contains the full reference.
    """
    df = pd.read_excel(path, dtype=str)
    if df.empty:
        return []
    col = df.columns[0]
    refs = df[col].dropna().astype(str).str.strip().tolist()
    refs = [r for r in refs if r and r.lower() not in ("article number", "artikelnummer")]
    refs.sort(key=len, reverse=True)
    return refs


def validate_reference(ref: str, catalog: list[str]) -> bool:
    if not catalog:
        return True
    return ref in catalog
