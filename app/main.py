import os
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse, JSONResponse, Response

from .extractor import extract_pdf, extract_pdfs_combined
from .catalog import load_catalog
from .clients import load_clients
from .customer_codes import load_customer_codes

app = FastAPI(title="Hoffmann PDF Extractor", version="4.0.0")

API_KEY = os.environ.get("API_KEY", "changeme")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

MAX_FILES_PER_REQUEST = 5

catalog: list[dict] = []
clients: list[dict] = []
customer_codes_map: dict = {}


@app.on_event("startup")
async def startup():
    global catalog, clients, customer_codes_map
    catalog_path = os.environ.get("CATALOG_PATH", "data/catalog.xlsx")
    clients_path = os.environ.get("CLIENTS_PATH", "data/clients.xlsx")
    codes_path = os.environ.get(
        "CUSTOMER_CODES_PATH", "data/codigos_referencias_clientes.xlsx"
    )
    try:
        catalog = load_catalog(catalog_path)
        print(f"[startup] Catalog loaded: {len(catalog)} references")
    except Exception as e:
        print(f"[startup] WARNING: could not load catalog: {e}")
        catalog = []
    try:
        clients = load_clients(clients_path)
        print(f"[startup] Clients loaded: {len(clients)} entries")
    except Exception as e:
        print(f"[startup] WARNING: could not load clients: {e}")
        clients = []
    try:
        customer_codes_map = load_customer_codes(codes_path)
        print(f"[startup] Customer codes loaded: {len(customer_codes_map)} mappings")
    except Exception as e:
        print(f"[startup] WARNING: could not load customer codes: {e}")
        customer_codes_map = {}


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


@app.get("/")
def root():
    return {"service": "Hoffmann PDF Extractor", "version": "4.0.0", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "catalog_size": len(catalog),
        "clients_size": len(clients),
        "customer_codes_size": len(customer_codes_map),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "max_files_per_request": MAX_FILES_PER_REQUEST,
    }


@app.get("/diag")
def diagnostic(_: str = Security(verify_api_key)):
    """Diagnostic endpoint: verifies OpenAI connectivity end-to-end."""
    result = {
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_key_prefix": (
            os.environ.get("OPENAI_API_KEY", "")[:10] + "..."
            if os.environ.get("OPENAI_API_KEY") else None
        ),
        "catalog_size": len(catalog),
        "clients_size": len(clients),
        "customer_codes_size": len(customer_codes_map),
    }
    if result["openai_key_set"]:
        try:
            from .llm import extract_with_llm
            test = extract_with_llm(
                "Purchase order number: TEST-123\n"
                "Date: 01/01/2026\n"
                "Buyer: Test User (test@example.com)\n"
                "Item: 759800 - Test article, 2 pcs, 10,00 EUR each, total 20,00 EUR"
            )
            result["openai_test"] = "ok"
            result["openai_response"] = test
        except Exception as e:
            result["openai_test"] = "failed"
            result["openai_error"] = f"{type(e).__name__}: {e}"
    return result


def _make_error_json(msg: str, status: int = 422):
    return JSONResponse(
        content={
            "status": "error",
            "error": msg,
            "orderNumber": "",
            "deliveryDate": "",
            "customerName": "",
            "deliveryAddress": "",
            "shippingCustomerNumber": "",
            "country": "",
            "contact": {"name": "", "email": "", "phone": ""},
            "partialExtraction": {"lines": []},
            "rawTextPreview": "",
        },
        status_code=status,
    )


def _safe_header(value: str) -> str:
    """HTTP headers cannot contain non-ASCII or control characters. Sanitize."""
    if not value:
        return ""
    s = str(value).encode("ascii", "ignore").decode("ascii")
    s = "".join(c for c in s if c.isprintable())
    return s.strip()[:200]  # keep it short


@app.post("/extract")
async def extract(
    file: List[UploadFile] = File(...),
    _: str = Security(verify_api_key),
):
    """
    Extract a purchase order from one OR SEVERAL PDFs.

    Responses:
    - HTTP 200, text/plain → success, body is HEAD|...\\r\\nLINE|...
      Header `X-Customer-Name` includes the customer's company name.
    - HTTP 422, application/json with status="ignored" → not a purchase order, skip silently.
    - HTTP 422, application/json with status="error" → could not extract, manual review needed.
    - HTTP 500 → unexpected server error.
    """
    files = file if isinstance(file, list) else [file]

    if not files:
        return _make_error_json("No files received", status=400)

    if len(files) > MAX_FILES_PER_REQUEST:
        return _make_error_json(
            f"Too many files ({len(files)}). Maximum allowed: {MAX_FILES_PER_REQUEST}.",
            status=400,
        )

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            return _make_error_json(
                f"File '{f.filename}' is not a PDF. Only PDFs are accepted.",
                status=400,
            )

    tmp_paths = []
    try:
        for f in files:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(await f.read())
                tmp_paths.append(tmp.name)

        if len(tmp_paths) == 1:
            success, result = extract_pdf(
                tmp_paths[0], catalog, clients, customer_codes_map=customer_codes_map
            )
        else:
            success, result = extract_pdfs_combined(
                tmp_paths, catalog, clients, customer_codes_map=customer_codes_map
            )

        if success:
            # result is a dict {text, customer_name, order_number}
            text_body = result.get("text", "") if isinstance(result, dict) else str(result)
            customer_name = result.get("customer_name", "") if isinstance(result, dict) else ""
            order_number = result.get("order_number", "") if isinstance(result, dict) else ""
            headers = {
                "X-Customer-Name": _safe_header(customer_name),
                "X-Order-Number": _safe_header(order_number),
            }
            return PlainTextResponse(
                content=text_body,
                status_code=200,
                headers=headers,
            )
        else:
            # Error or ignored — return JSON
            return JSONResponse(content=result, status_code=422)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _make_error_json(
            f"Unexpected server error: {type(e).__name__}: {e}",
            status=500,
        )
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
