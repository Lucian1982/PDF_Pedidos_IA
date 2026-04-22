# Hoffmann PDF Extractor

API REST que recibe pedidos de compra en PDF (de cualquier cliente/formato) y los convierte al formato pipe delimitado de Hoffmann.

## Formato de salida

```
HEAD|<nº_cliente>|<nº_pedido>|<comprador>|<fecha>||||| <ES|PT>||<email>
LINE|<ref_hoffmann>|<cantidad>||<precio_unit>|<importe>
LINE|...
```

## Archivos de datos necesarios

Coloca estos dos archivos en la carpeta `data/` antes de desplegar:

| Archivo | Descripción |
|---|---|
| `data/catalog.xlsx` | Catálogo Hoffmann. La primera columna debe ser `Artikelnummer` con las referencias completas (ej. `759800`, `642229 8`, `082812 M12`) |
| `data/clients.xlsx` | Tabla de clientes con columnas: `Direccion envio`, `Numero clinete`, `Pais` |

## Variables de entorno

Configura estas variables en el dashboard de Render (nunca en el código):

| Variable | Descripción |
|---|---|
| `API_KEY` | Clave secreta que Power Automate enviará en el header `X-API-Key` |
| `OPENAI_API_KEY` | Tu clave de OpenAI (usada como fallback para PDFs complejos) |
| `CATALOG_PATH` | Ruta al catálogo (por defecto: `data/catalog.xlsx`) |
| `CLIENTS_PATH` | Ruta a la tabla de clientes (por defecto: `data/clients.xlsx`) |

## Despliegue en Render

1. Haz fork o clona este repo en tu GitHub
2. Copia tus archivos Excel en `data/`
3. Haz commit y push
4. En [render.com](https://render.com): New → Web Service → conecta tu repo
5. Render detecta el `render.yaml` automáticamente
6. En Environment Variables del dashboard, añade `API_KEY` y `OPENAI_API_KEY`
7. Deploy → en 2-3 minutos tienes la URL pública

## Uso desde Power Automate

Acción: **HTTP**

- Method: `POST`
- URL: `https://tu-servicio.onrender.com/extract`
- Headers:
  - `X-API-Key`: tu API key
- Body: `multipart/form-data` con el PDF en el campo `file`

La respuesta es texto plano con el formato HEAD/LINE.

## Probar localmente

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar variables
export API_KEY=test123
export OPENAI_API_KEY=sk-...

# Arrancar servidor
uvicorn app.main:app --reload

# Probar (en otro terminal)
curl -X POST http://localhost:8000/extract \
  -H "X-API-Key: test123" \
  -F "file=@pedido.pdf"
```

También puedes abrir http://localhost:8000/docs para ver la interfaz interactiva de FastAPI.

## Lógica de extracción

```
PDF → pdfplumber (texto + tablas)
        ↓
     Cabecera: regex → si falta algo → OpenAI gpt-4o-mini
        ↓
     Dirección de entrega → match por CP + fuzzy → número de cliente + ES/PT
        ↓
     Líneas: tabla pdfplumber → lookup inverso en catálogo (112k refs)
        ↓
     Si no encuentra referencia → OpenAI gpt-4o-mini por línea
        ↓
     Validación: precio × cantidad = importe
        ↓
     Salida HEAD|...\r\nLINE|...
```

## Estructura del proyecto

```
pdf-extractor/
├── app/
│   ├── main.py        # FastAPI + endpoint /extract
│   ├── extractor.py   # Lógica principal de extracción
│   ├── catalog.py     # Carga y búsqueda en catálogo Hoffmann
│   └── clients.py     # Carga y match de clientes por dirección
├── data/
│   ├── catalog.xlsx   # (NO subir a GitHub si es confidencial)
│   └── clients.xlsx   # (NO subir a GitHub si es confidencial)
├── requirements.txt
├── render.yaml
└── README.md
```

> **Nota de privacidad**: Si el catálogo o la tabla de clientes son confidenciales, añade `data/` al `.gitignore` y sube los archivos directamente en el servidor de Render usando su interfaz de ficheros, o usa un bucket S3/Azure Blob.
