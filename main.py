import os
import base64
import io
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from doclayout_yolo import YOLOv10
import cv2
import numpy as np

# CONFIGURACIÓN Y LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Variables globales (los modelos se cargan al inicio)
MODEL_LAYOUT = None
MODEL_OCR = None
PROCESSOR_OCR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parámetros configurables (pueden venir de variables de entorno)
ANCHO_REFERENCIA = int(os.getenv("HTML_WIDTH", "800"))
TOLERANCIA_VERTICAL = int(os.getenv("TOLERANCIA_VERTICAL", "20"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.2"))

# Mapeo de clases de DocLayout‑YOLO
CLASE_NOMBRE = {
    0: "text",
    1: "title",
    2: "list",
    3: "table",
    4: "figure"
}

# MODELOS DE DATOS (Pydantic)
class ProcessRequest(BaseModel):
    """
    Petición para el endpoint /process.
    """
    image: str = Field(..., description="Imagen codificada en base64 (puede incluir prefijo data:...)")
    filename: Optional[str] = Field("documento", description="Nombre opcional del archivo")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Opciones adicionales (reservado)")

class ProcessResponse(BaseModel):
    """
    Respuesta del endpoint /process.
    """
    html: str = Field(..., description="HTML estructurado con el contenido del documento")
    markdown: Optional[str] = Field(None, description="Versión en Markdown (opcional)")
    blocks: List[Dict[str, Any]] = Field(default_factory=list, description="Bloques detectados con coordenadas y texto")
    metadata: Dict[str, Any] = Field(..., description="Metadatos: dimensiones, número de bloques, etc.")

# FUNCIONES DE CARGA DE MODELOS
def load_models() -> bool:
    """
    Carga los modelos en memoria (DocLayout‑YOLO y TrOCR).
    Se ejecuta al iniciar el servicio.
    """
    global MODEL_LAYOUT, MODEL_OCR, PROCESSOR_OCR, DEVICE
    try:
        logger.info(f"Cargando modelos en dispositivo: {DEVICE}")
        
        # DocLayout‑YOLO (detección de layout)
        MODEL_LAYOUT = YOLOv10("juliozhao/DocLayout-YOLO-DocStructBench")
        logger.info("DocLayout‑YOLO cargado correctamente")
        
        # TrOCR (reconocimiento de texto)
        PROCESSOR_OCR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        MODEL_OCR = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        MODEL_OCR.to(DEVICE)
        MODEL_OCR.eval()
        logger.info("TrOCR cargado correctamente")
        
        return True
    except Exception as e:
        logger.error(f"Error al cargar modelos: {e}")
        return False

# FUNCIONES DE PROCESAMIENTO
def decode_image(base64_str: str) -> Image.Image:
    """
    Decodifica una imagen en base64 a un objeto PIL Image.
    Soporta prefijos como "data:image/jpeg;base64,...".
    """
    if ";" in base64_str and "," in base64_str:
        base64_str = base64_str.split(",")[1]
    try:
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error al decodificar la imagen base64: {e}")

def extract_text_trocr(image_pil: Image.Image, x1: int, y1: int, x2: int, y2: int) -> str:
    """
    Extrae texto de una región de la imagen usando TrOCR.
    """
    if x2 <= x1 or y2 <= y1:
        return ""
    region = image_pil.crop((x1, y1, x2, y2))
    if region.width == 0 or region.height == 0:
        return ""
    
    # Procesar con TrOCR
    pixel_values = PROCESSOR_OCR(region, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(DEVICE)
    with torch.no_grad():
        generated_ids = MODEL_OCR.generate(pixel_values)
    text = PROCESSOR_OCR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def order_blocks(boxes, coords_list: List[List[float]], texts: List[str]) -> List[Dict]:
    """
    Ordena los bloques en orden de lectura:
      1. Primero por coordenada Y (de arriba a abajo)
      2. Dentro de la misma franja horizontal, por X (de izquierda a derecha)
    """
    bloques = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = coords_list[i]
        bloques.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "texto": texts[i],
            "tipo": int(box.cls[0].item())
        })
    
    # Orden principal por Y (top)
    bloques.sort(key=lambda b: b["y1"])
    
    # Reordenar dentro de cada fila (misma franja vertical)
    resultado = []
    fila_actual = []
    ultimo_y = None
    for bloque in bloques:
        if ultimo_y is None or abs(bloque["y1"] - ultimo_y) <= TOLERANCIA_VERTICAL:
            fila_actual.append(bloque)
        else:
            fila_actual.sort(key=lambda b: b["x1"])
            resultado.extend(fila_actual)
            fila_actual = [bloque]
        ultimo_y = bloque["y1"]
    
    if fila_actual:
        fila_actual.sort(key=lambda b: b["x1"])
        resultado.extend(fila_actual)
    
    return resultado

def generate_html(blocks: List[Dict], img_w: int, img_h: int, ancho_salida: int = ANCHO_REFERENCIA) -> str:
    """
    Genera un HTML con posicionamiento absoluto que replica el diseño original.
    """
    escala = ancho_salida / img_w
    alto_salida = img_h * escala
    
    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Documento Reconstruido</title>
<style>
    body {{
        margin: 0;
        padding: 20px;
        background: #f5f5f5;
        font-family: 'Segoe UI', Arial, sans-serif;
        width: {ancho_salida}px;
    }}
    .document {{
        position: relative;
        width: {ancho_salida}px;
        min-height: {alto_salida}px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0 auto;
        padding: 10px;
        box-sizing: border-box;
    }}
    .block {{
        position: absolute;
        overflow: hidden;
        white-space: normal;
        word-wrap: break-word;
    }}
    .title {{ font-size: 1.8em; font-weight: bold; margin: 0; }}
    .text {{ font-size: 1em; margin: 0; line-height: 1.5; }}
    .list {{ font-size: 1em; margin: 0 0 0 20px; }}
    .table {{ border-collapse: collapse; width: 100%; }}
    .table td, .table th {{ border: 1px solid #ccc; padding: 4px; }}
    .figure {{ background: #f0f0f0; text-align: center; font-style: italic; padding: 10px; }}
</style>
</head>
<body>
<div class="document">
"""]
    
    for bloque in blocks:
        x1, y1, x2, y2 = bloque["x1"], bloque["y1"], bloque["x2"], bloque["y2"]
        texto = bloque["texto"].replace("<", "&lt;").replace(">", "&gt;")
        clase_id = bloque["tipo"]
        clase_nombre = CLASE_NOMBRE.get(clase_id, "text")
        
        left = x1 * escala
        top = y1 * escala
        width = (x2 - x1) * escala
        height = (y2 - y1) * escala
        style = f"left: {left}px; top: {top}px; width: {width}px; height: {height}px;"
        
        if clase_nombre == "title":
            html_parts.append(f'<div class="block title" style="{style}"><h1>{texto}</h1></div>')
        elif clase_nombre == "list":
            items = [it.strip() for it in texto.split("\n") if it.strip()]
            if items:
                lista_html = "<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>"
                html_parts.append(f'<div class="block list" style="{style}">{lista_html}</div>')
            else:
                html_parts.append(f'<div class="block text" style="{style}"><p>{texto}</p></div>')
        elif clase_nombre == "table":
            # Mejorable: intentar parsear tablas con más estructura
            html_parts.append(f'<div class="block table" style="{style}"><pre>{texto}</pre></div>')
        elif clase_nombre == "figure":
            html_parts.append(f'<div class="block figure" style="{style}">[Imagen detectada]</div>')
        else:
            html_parts.append(f'<div class="block text" style="{style}"><p>{texto}</p></div>')
    
    html_parts.append("</div></body></html>")
    return "\n".join(html_parts)

def process_single_image(image_base64: str) -> Dict[str, Any]:
    """
    Pipeline completo para una imagen:
      1. Decodificar
      2. Detectar layout (DocLayout‑YOLO)
      3. Extraer texto por bloque (TrOCR)
      4. Ordenar bloques
      5. Generar HTML
    """
    # 1. Decodificar
    imagen_pil = decode_image(image_base64)
    img_w, img_h = imagen_pil.size
    logger.info(f"📄 Imagen procesada: {img_w}x{img_h} px")
    
    # 2. Detección de layout
    # Guardar temporalmente para YOLO (acepta rutas)
    temp_path = "/tmp/temp_doc.jpg"
    imagen_pil.save(temp_path)
    
    results = MODEL_LAYOUT.predict(temp_path, imgsz=1024, conf=CONFIDENCE_THRESHOLD, device=DEVICE)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        logger.warning("⚠️ No se detectaron bloques de texto")
        return {
            "html": "<p>No se detectaron bloques de texto en la imagen.</p>",
            "blocks": [],
            "metadata": {"width": img_w, "height": img_h, "num_blocks": 0}
        }
    
    # 3. Extraer texto con TrOCR para cada bloque
    coordenadas = []
    textos = []
    for box in boxes:
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        coordenadas.append(coords)
        texto = extract_text_trocr(imagen_pil, *coords)
        textos.append(texto)
    
    # 4. Ordenar bloques
    bloques_ordenados = order_blocks(boxes, coordenadas, textos)
    
    # 5. Generar HTML
    html_final = generate_html(bloques_ordenados, img_w, img_h)
    
    # 6. Generar Markdown (versión simplificada)
    markdown_lines = []
    for b in bloques_ordenados:
        clase = CLASE_NOMBRE.get(b["tipo"], "text")
        txt = b["texto"]
        if clase == "title":
            markdown_lines.append(f"# {txt}")
        elif clase == "list":
            for item in txt.split("\n"):
                if item.strip():
                    markdown_lines.append(f"- {item.strip()}")
        else:
            markdown_lines.append(txt)
    markdown = "\n\n".join(markdown_lines)
    
    return {
        "html": html_final,
        "markdown": markdown,
        "blocks": bloques_ordenados,
        "metadata": {
            "width": img_w,
            "height": img_h,
            "num_blocks": len(bloques_ordenados),
            "device": DEVICE
        }
    }

# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================
app = FastAPI(
    title="OCR Layout Service",
    description="Microservicio para detección de layout y OCR de documentos usando DocLayout‑YOLO y TrOCR.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Carga los modelos al iniciar el servicio."""
    success = load_models()
    if not success:
        logger.error("No se pudieron cargar los modelos. El servicio puede fallar.")
    else:
        logger.info("Servicio listo para recibir peticiones")

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Endpoint de salud. Devuelve el estado del servicio y los modelos.
    """
    models_loaded = MODEL_LAYOUT is not None and MODEL_OCR is not None
    return {
        "status": "ok" if models_loaded else "degraded",
        "device": DEVICE,
        "models_loaded": models_loaded,
        "html_width": ANCHO_REFERENCIA,
        "tolerance_vertical": TOLERANCIA_VERTICAL
    }

@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Procesa una imagen y devuelve HTML estructurado.
    
    - **image**: imagen en base64 (puede incluir prefijo data:...)
    - **filename**: nombre opcional del archivo
    - **options**: opciones adicionales (reservado para futuras extensiones)
    """
    if MODEL_LAYOUT is None or MODEL_OCR is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados aún. Intente más tarde.")
    
    try:
        result = process_single_image(request.image)
        # Añadir el filename a los metadatos
        result["metadata"]["filename"] = request.filename
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error procesando documento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/process_batch")
async def process_batch(requests: List[ProcessRequest]):
    """
    Procesa múltiples imágenes (lote).
    """
    if MODEL_LAYOUT is None or MODEL_OCR is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados aún.")
    
    resultados = []
    for req in requests:
        try:
            res = process_single_image(req.image)
            res["metadata"]["filename"] = req.filename
            resultados.append(res)
        except Exception as e:
            resultados.append({
                "error": str(e),
                "filename": req.filename
            })
    return {"results": resultados}

# PUNTO DE ENTRADA PARA DESARROLLO
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )