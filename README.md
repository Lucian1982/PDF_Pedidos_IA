# Hoffmann PDF Extractor

API REST que recibe pedidos de compra en PDF (de cualquier cliente/formato) y los convierte al formato pipe delimitado de Hoffmann.

## Arquitectura híbrida

```
PDF → pdfplumber → reglas (Aludium, Astemo, genérico)
                         ↓
                   ¿calidad OK?
                   ├── Sí → devolver resultado (0€, instantáneo)
                   └── No → OpenAI gpt-4o-mini (~0.001€, 3s)
                         ↓
                   Validar referencias contra catálogo
                         ↓
                   Match cliente por dirección
                         ↓
                   Salida HEAD|LINE|...
```

Beneficios:
- **Clientes conocidos** (Aludium, Astemo, ...): coste 0€, respuesta inmediata
- **Clientes nuevos/formatos raros**: OpenAI actúa de red de seguridad
- **Cobertura**: 100% desde el día 1, escala a cientos de clientes

## Criterios de calidad que disparan el fallback

Las reglas se consideran insuficientes (y se llama a OpenAI) si:
- No se encuentra `orderNumber`
- No se extrae ninguna línea
- Alguna línea no tiene referencia, cantidad o precio
- `cantidad × precio_unit` no cuadra con el `importe` (tolerancia 2%)

## Formato de salida

```
HEAD|<nº_cliente>|<nº_pedido>|<comprador>|<fecha>|||||<ES|PT>||<email>
LINE|<ref_hoffmann>|<cantidad>||<precio_unit>|<importe>
LINE|...
```

## Archivos de datos

Coloca estos dos archivos en `data/`:

| Archivo | Descripción |
|---|---|
| `data/catalog.xlsx` | Catálogo Hoffmann. Primera columna = `Artikelnummer` |
| `data/clients.xlsx` | Columnas: `Direccion envio`, `Numero clinete`, `Pais` |

## Variables de entorno (en Render)

| Variable | Requerida | Descripción |
|---|---|---|
| `API_KEY` | Sí | Clave para autenticar peticiones de Power Automate |
| `OPENAI_API_KEY` | Recomendada | Sin ella, solo funcionan las reglas; con ella, se cubre todo |
| `CATALOG_PATH` | No | Por defecto `data/catalog.xlsx` |
| `CLIENTS_PATH` | No | Por defecto `data/clients.xlsx` |

## Despliegue en Render

1. Sube el repo a GitHub
2. En [render.com](https://render.com): **New → Web Service** → conecta tu repo
3. Render detecta `render.yaml` automáticamente
4. En **Environment Variables** añade `API_KEY` y `OPENAI_API_KEY`
5. Click **Deploy** → en 2-3 minutos tienes la URL pública

## Uso desde Power Automate

Acción **HTTP** (premium):

- Method: `POST`
- URL: `https://tu-servicio.onrender.com/extract`
- Headers: `X-API-Key: tu_api_key`
- Body: `multipart/form-data`, campo `file` con el PDF

La respuesta es texto plano con el formato HEAD/LINE.

## Endpoints

| Ruta | Descripción |
|---|---|
| `GET /` | Info básica del servicio |
| `GET /health` | Estado y tamaño del catálogo/clientes cargados |
| `GET /docs` | Documentación interactiva Swagger |
| `POST /extract` | Procesa un PDF y devuelve HEAD/LINE (requiere API Key) |

## Probar localmente

```bash
pip install -r requirements.txt
export API_KEY=test123
export OPENAI_API_KEY=sk-...   # opcional, para fallback
uvicorn app.main:app --reload
```

Abre http://localhost:8000/docs para probar.

## Estructura

```
pdf-extractor/
├── app/
│   ├── __init__.py
│   ├── main.py          # API FastAPI
│   ├── extractor.py     # Lógica principal (reglas + orquestación)
│   ├── catalog.py       # Carga del catálogo Hoffmann
│   ├── clients.py       # Match de clientes por dirección
│   └── llm.py           # Fallback OpenAI
├── data/
│   ├── catalog.xlsx     # Tu catálogo real
│   └── clients.xlsx     # Tu tabla de clientes
├── requirements.txt
├── render.yaml
└── README.md
```

## Coste estimado

Para 100 PDFs/día:
- Render plan Starter: ~7$/mes
- OpenAI (suponiendo 10% de PDFs caen al fallback): ~1$/mes
- **Total: ~8$/mes**
