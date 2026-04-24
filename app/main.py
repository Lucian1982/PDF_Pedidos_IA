import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse, JSONResponse

from .extractor import extract_pdf
from .catalog import load_catalog
from .clients import load_clients

app = FastAPI(title="Hoffmann PDF Extractor", version="2.0.0")

API_KEY = os.environ.get("API_KEY", "changeme")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

catalog: list[str] = []
clients: list[dict] = []


@app.on_event("startup")
async def startup():
    global catalog, clients
    catalog_path = os.environ.get("CATALOG_PATH", "data/catalog.xlsx")
    clients_path = os.environ.get("CLIENTS_PATH", "data/clients.xlsx")
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


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key


@app.get("/")
def root():
    return {"service": "Hoffmann PDF Extractor", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "catalog_size": len(catalog),
        "clients_size": len(clients),
        "openai_configured": bool(os.environ.get("OPENAI_API_KEY")),
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


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    _: str = Security(verify_api_key),
):
    """
    Extract a purchase order PDF.
    - On success: returns text/plain with HEAD|...\\r\\nLINE|... format (HTTP 200).
    - On error:   returns application/json with error details (HTTP 422).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        success, result = extract_pdf(tmp_path, catalog, clients)
        if success:
            return PlainTextResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=422)
    except Exception as e:
        # Unexpected crash → return JSON error
        return JSONResponse(
            content={
                "status": "error",
                "error": f"Unexpected server error: {type(e).__name__}: {e}",
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
            status_code=500,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
