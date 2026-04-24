import pandas as pd
from rapidfuzz import fuzz, process


def load_catalog(path: str) -> list[dict]:
    """
    Load the Hoffmann catalog.
    Returns list of dicts: {'ref': '...', 'name': '...'}.
    'name' combines the short description, brand and product name for
    fuzzy-matching.
    """
    df = pd.read_excel(path, dtype=str)
    if df.empty:
        return []

    cols = df.columns.tolist()
    ref_col = cols[0]
    desc_col = brand_col = pname_col = None
    for c in cols:
        cl = c.lower()
        if desc_col is None and ("kurzbeschreibung" in cl or "short description" in cl
                                 or "descripci" in cl):
            desc_col = c
        if brand_col is None and ("marke" in cl or "brand" in cl or "marca" in cl):
            brand_col = c
        if pname_col is None and ("produktname" in cl or "product name" in cl
                                  or "nombre" in cl):
            pname_col = c

    out = []
    for _, row in df.iterrows():
        ref = str(row[ref_col] or "").strip()
        if not ref or ref.lower() in ("article number", "artikelnummer"):
            continue

        parts = []
        for col in (desc_col, brand_col, pname_col):
            if col:
                v = str(row[col] or "").strip()
                if v and v.lower() != "nan":
                    parts.append(v)
        name = " | ".join(parts)
        out.append({"ref": ref, "name": name})

    out.sort(key=lambda x: len(x["ref"]), reverse=True)
    return out


def validate_reference(ref: str, catalog: list[dict]) -> bool:
    if not catalog:
        return True
    return any(c["ref"] == ref for c in catalog)


def find_reference_by_name(description: str, catalog: list[dict],
                            min_confidence: int = 90) -> tuple[str, float]:
    """
    Fuzzy-match the given description against catalog product names.
    Returns (ref, score). If no match above min_confidence, returns ('', score_of_best).
    """
    if not description or not catalog:
        return ("", 0.0)

    names = [c["name"] for c in catalog]
    result = process.extractOne(
        description,
        names,
        scorer=fuzz.token_set_ratio,
        score_cutoff=min_confidence,
    )
    if result:
        _, score, idx = result
        return (catalog[idx]["ref"], float(score))

    # Report best score even if below threshold
    result_any = process.extractOne(
        description,
        names,
        scorer=fuzz.token_set_ratio,
    )
    if result_any:
        return ("", float(result_any[1]))
    return ("", 0.0)
