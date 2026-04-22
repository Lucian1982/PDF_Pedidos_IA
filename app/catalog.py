import pandas as pd


def load_catalog(path: str) -> list[str]:
    """
    Loads the Hoffmann catalog.
    First column (Artikelnummer) contains the full reference,
    e.g. '759800', '759856 600', '082812 M12', '640190 1/2'.
    """
    df = pd.read_excel(path, dtype=str)
    if df.empty:
        return []
    col = df.columns[0]
    refs = df[col].dropna().astype(str).str.strip().tolist()
    # Skip header-like rows
    refs = [r for r in refs if r and r.lower() not in ("article number", "artikelnummer")]
    # Sort longest first so "642229 8" matches before "642229"
    refs.sort(key=len, reverse=True)
    return refs


def validate_reference(ref: str, catalog: list[str]) -> bool:
    """Returns True if the reference exists in the catalog."""
    if not catalog:
        return True  # no catalog loaded -> don't block
    return ref in catalog
