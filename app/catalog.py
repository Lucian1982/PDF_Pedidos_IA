import pandas as pd
import re


def load_catalog(path: str) -> list[str]:
    df = pd.read_excel(path, dtype=str)
    col = df.columns[0]
    refs = df[col].dropna().str.strip().tolist()
    refs.sort(key=len, reverse=True)
    return refs


def find_reference(text: str, catalog: list[str]) -> str | None:
    text_upper = text.upper()
    for ref in catalog:
        ref_upper = ref.upper()
        pattern = r'(?<![A-Z0-9])' + re.escape(ref_upper) + r'(?![A-Z0-9/])'
        if re.search(pattern, text_upper):
            return ref

    article_pattern = r'(?:art[ií]culo|article|item|n[r°º]\.?|num\.?)\s*:?\s*([A-Z0-9]+(?:\s+[A-Z0-9/]+)?)'
    match = re.search(article_pattern, text_upper)
    if match:
        candidate = match.group(1).strip()
        for ref in catalog:
            if ref.upper() == candidate:
                return ref

    return None
