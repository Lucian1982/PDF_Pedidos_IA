import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse

from .extractor import extract_pdf
from .catalog import load_catalog
from .clients import load_clients

app = FastAPI(title="Hoffmann PDF Extractor", version="1.0.0")

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
    return {"service": "Hoffmann PDF Extractor", "status": "running"}


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
    """Diagnostic endpoint to verify OpenAI connectivity. Requires API Key."""
    result = {
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_key_prefix": (os.environ.get("OPENAI_API_KEY", "")[:10] + "...") if os.environ.get("OPENAI_API_KEY") else None,
        "catalog_size": len(catalog),
        "clients_size": len(clients),
    }
    if result["openai_key_set"]:
        try:
            from .llm import extract_with_llm
            test = extract_with_llm("Test PDF: Purchase order number: TEST123\nDate: 01/01/2026\nBuyer: Test User")
            result["openai_test"] = "ok"
            result["openai_response"] = test
        except Exception as e:
            result["openai_test"] = "failed"
            result["openai_error"] = f"{type(e).__name__}: {e}"
    return result


@app.post("/extract", response_class=PlainTextResponse)
async def extract(
    file: UploadFile = File(...),
    _: str = Security(verify_api_key),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return extract_pdf(tmp_path, catalog, clients)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
