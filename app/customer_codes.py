"""
Customer-specific code → Hoffmann-reference lookup.

Some customers use their own internal codes when ordering (e.g. Continental
orders with code '00080218600' which corresponds to Hoffmann ref '626089 9').
This module loads a separate Excel that maps those customer codes to Hoffmann
references, and provides a flexible lookup function.

Excel columns: VAT/NIF/CIF | Codigo Cliente | Codigo Hoffmann
"""

from __future__ import annotations
import re
import pandas as pd


def load_customer_codes(path: str) -> dict:
    """
    Load the customer-codes Excel.
    Returns a dict keyed by (normalized_vat, normalized_customer_code) → hoffmann_ref.

    Returns empty dict if file does not exist or is empty.
    """
    try:
        df = pd.read_excel(path, dtype=str)
    except (FileNotFoundError, OSError):
        print(f"[customer_codes] No file at {path}, lookups disabled")
        return {}

    if df.empty:
        return {}

    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if ("vat" in cl or "nif" in cl or "cif" in cl):
            col_map["vat"] = c
        elif "codigo" in cl and "cliente" in cl:
            col_map["customer_code"] = c
        elif "codigo" in cl and ("hoffmann" in cl or "hoff" in cl):
            col_map["hoffmann_ref"] = c

    if not all(k in col_map for k in ("vat", "customer_code", "hoffmann_ref")):
        print(f"[customer_codes] Missing required columns in {path}. "
              f"Expected: VAT/NIF/CIF, Codigo Cliente, Codigo Hoffmann. Got: {list(df.columns)}")
        return {}

    mapping = {}
    for _, row in df.iterrows():
        vat = str(row.get(col_map["vat"], "")).strip()
        cust_code = str(row.get(col_map["customer_code"], "")).strip()
        hoff_ref = str(row.get(col_map["hoffmann_ref"], "")).strip()
        if not vat or not cust_code or not hoff_ref:
            continue
        key = (_normalize_vat(vat), _normalize_code(cust_code))
        mapping[key] = hoff_ref

    print(f"[customer_codes] Loaded {len(mapping)} customer-code mappings")
    return mapping


def _normalize_vat(vat: str) -> str:
    """Normalize VAT: strip whitespace, uppercase."""
    return re.sub(r"\s+", "", vat).upper()


def _normalize_code(code: str) -> str:
    """
    Normalize a customer code for flexible matching.
    Strip whitespace, dashes, dots, commas. Uppercase.
    Examples:
      '00080218600'  → '00080218600'
      '0008021-8600' → '00080218600'
      '0008021,8600' → '00080218600'
      '576705 W'     → '576705W'
      '576705W'      → '576705W'
    """
    if not code:
        return ""
    return re.sub(r"[\s\-.,/]", "", code).upper()


def find_hoffmann_for_customer_code(
    customer_vat: str,
    customer_code: str,
    mapping: dict,
) -> str:
    """
    Return the Hoffmann reference for a (vat, customer_code) pair, or empty
    string if not found.
    """
    if not customer_vat or not customer_code or not mapping:
        return ""

    vat_n = _normalize_vat(customer_vat)
    code_n = _normalize_code(customer_code)
    if not vat_n or not code_n:
        return ""

    return mapping.get((vat_n, code_n), "")
