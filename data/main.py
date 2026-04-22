import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import PlainTextResponse
import tempfile

from .extractor import extract_pdf
from .catalog import load_catalog
from .clients import load_clients

app = FastAPI(title="Hoffmann PDF Extractor", version="1.0.0")

API_KEY = os.environ.get("API_KEY", "changeme")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Load catalog and clients at startup
catalog = None
clients = None

@app.on_event("startup")
async def startup():
    global catalog, clients
    catalog_path = os.environ.get("CATALOG_PATH", "data/catalog.xlsx")
    clients_path = os.environ.get("CLIENTS_PATH", "data/clients.xlsx")
    catalog = load_catalog(catalog_path)
    clients = load_clients(clients_path)
    print(f"Catalog loaded: {len(catalog)} references")
    print(f"Clients loaded: {len(clients)} entries")

def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": len(catalog) if catalog else 0}

@app.post("/extract", response_class=PlainTextResponse)
async def extract(
    file: UploadFile = File(...),
    _: str = Security(verify_api_key)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = extract_pdf(tmp_path, catalog, clients)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

