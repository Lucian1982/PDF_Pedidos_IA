import pandas as pd
import re


def load_catalog(path: str) -> list[str]:
    """
    Loads the Hoffmann catalog and returns a list of article numbers (Artikelnummer),
    sorted by length descending so longer/more specific references match first.
    """
    df = pd.read_excel(path, dtype=str)

    # The first column is Artikelnummer (e.g. "082812 M12", "759800", "642229 8")
    col = df.columns[0]
    refs = df[col].dropna().str.strip().tolist()

    # Sort longest first so "642229 8" matches before "642229"
    refs.sort(key=len, reverse=True)
    return refs


def find_reference(text: str, catalog: list[str]) -> str | None:
    """
    Searches the description text for a catalog reference.
    Returns the first (longest) match found, or None.
    """
    text_upper = text.upper()
    for ref in catalog:
        ref_upper = ref.upper()
        # Match the reference as a whole word/token to avoid partial matches
        pattern = r'(?<![A-Z0-9])' + re.escape(ref_upper) + r'(?![A-Z0-9/])'
        if re.search(pattern, text_upper):
            return ref

    # Second pass: try matching just numeric parts for cases like
    # "Número de artículo: 845020 18" where ref is "845020 18"
    # Look for pattern "artículo|article|item|n.º|nr." followed by the ref
    article_pattern = r'(?:art[ií]culo|article|item|n[r°º]\.?|num\.?)\s*:?\s*([A-Z0-9]+(?:\s+[A-Z0-9/]+)?)'
    match = re.search(article_pattern, text_upper)
    if match:
        candidate = match.group(1).strip()
        # Check if this candidate exists in catalog
        for ref in catalog:
            if ref.upper() == candidate:
                return ref

    return None
