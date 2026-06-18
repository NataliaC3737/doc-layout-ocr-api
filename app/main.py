# -*- coding: utf-8 -*-
import os
import json
import logging
import time
import shutil
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from app.core_layout_segmentation import LocalLayoutVisionDetector, MultimodalStructureExtractor
from app.document_processing_utils import rasterize_document_to_rgb_images

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Enterprise Document Layout & Reconstruction API",
    version="1.0.0"
)

MODELO_YOLO_PATH = os.getenv("MODELO_YOLO_PATH", None)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)

layout_detector_engine = None
multimodal_extractor_engine = None

@app.on_event("startup")
def verify_and_initialize_engines():
    global layout_detector_engine, multimodal_extractor_engine
    if not GEMINI_API_KEY:
        logging.error("Falta la variable de entorno 'GEMINI_API_KEY'.")
    else:
        multimodal_extractor_engine = MultimodalStructureExtractor(google_api_key=GEMINI_API_KEY)
        
    try:
        layout_detector_engine = LocalLayoutVisionDetector(trained_model_path=MODELO_YOLO_PATH)
    except Exception as error:
        logging.critical(f"Error al cargar LocalLayoutVisionDetector: {error}")


@app.get("/health")
def health_check():
    """
    Verifica el estado de salud de la API y sus componentes.

    Returns:
        dict: Estado del servicio y de los modelos de IA cargados.
    """
    return {
        "status": "healthy", 
        "components": {
            "yolo_local": layout_detector_engine is not None, 
            "gemini_multimodal": multimodal_extractor_engine is not None
        }
    }


@app.post("/api/v1/convert-to-html", response_class=HTMLResponse)
async def process_document_pipeline(file: UploadFile = File(...)):
    """
    Convierte un archivo (PDF o Imagen) en un documento HTML estructurado.

    Args:
        file (UploadFile): Archivo enviado mediante formulario (Multipart).

    Returns:
        str: Código HTML completo con el texto e imágenes incrustadas.
    """
    global layout_detector_engine, multimodal_extractor_engine

    if not layout_detector_engine or not multimodal_extractor_engine:
        raise HTTPException(status_code=503, detail="Los motores de IA no están listos.")
    
    temporary_storage_path = f"temp_io_{int(time.time())}_{file.filename}"

    with open(temporary_storage_path, "wb") as disk_buffer:
        shutil.copyfileobj(file.file, disk_buffer)
        
    try:
        # 1. Convertir documento a imágenes
        document_pages = rasterize_document_to_rgb_images(temporary_storage_path)
        
        # 2. Crear documento HTML base
        master_html_soup = BeautifulSoup("<!DOCTYPE html><html><head><meta charset='utf-8'></head><body></body></html>", "html.parser")
        master_document_body = master_html_soup.body

        # 3. Procesar cada página
        for page_index, page_image in enumerate(document_pages):
            # YOLO: Detectar regiones visuales (Tablas, logos, firmas...)
            vision_analysis = layout_detector_engine.segment_document_layout(page_image)
            detected_regions = vision_analysis["regions"]

            # Gemini: Obtener el HTML con el texto estructurado
            gemini_json_string = multimodal_extractor_engine.convert_page_to_semantic_html(page_image, page_index=page_index)
            
            try:
                parsed_json_data = json.loads(gemini_json_string)
                extracted_html_fragment = parsed_json_data.get("html", "")
            except Exception:
                extracted_html_fragment = gemini_json_string

            page_soup_fragment = BeautifulSoup(extracted_html_fragment, "html.parser")

            # Clasificar las regiones encontradas por YOLO
            logos = [r for r in detected_regions if r["label"] == "Logo"]
            firmas = [r for r in detected_regions if r["label"] == "Firma"]
            sellos = [r for r in detected_regions if r["label"] == "Sello"]
            imagenes = [r for r in detected_regions if r["label"] == "Imagen"]

            # Reemplazar marcadores de imagen de Gemini por los recortes reales en Base64
            rendered_img_tags = page_soup_fragment.find_all("img")

            for img_tag in rendered_img_tags:
                source_attribute = img_tag.get("src", "")
                alt_attribute = img_tag.get("alt", "").lower()

                if f"ORIGINAL_IMAGE_{page_index}" in source_attribute or source_attribute == f"ORIGINAL_IMAGE_{page_index}":
                    assigned_visual_crop = None

                    if "logo" in alt_attribute and logos: assigned_visual_crop = logos.pop(0)
                    elif ("firma" in alt_attribute or "signature" in alt_attribute) and firmas: assigned_visual_crop = firmas.pop(0)
                    elif ("sello" in alt_attribute or "stamp" in alt_attribute) and sellos: assigned_visual_crop = sellos.pop(0)
                    elif imagenes: assigned_visual_crop = imagenes.pop(0)
                    elif detected_regions: assigned_visual_crop = detected_regions.pop(0)

                    if assigned_visual_crop:
                        img_tag["src"] = assigned_visual_crop["croppedBase64"]
                        img_tag["alt"] = f"YOLO {assigned_visual_crop['label']} (Pág {page_index+1})"
                        assigned_visual_crop["used"] = True

            # Inyectar firmas o sellos que Gemini omitió pero YOLO sí detectó
            unmapped_critical_regions = [r for r in detected_regions if r.get("label") in ["Firma", "Sello"] and not r.get("used", False)]
            
            for critical_item in unmapped_critical_regions:
                autonomous_img_tag = page_soup_fragment.new_tag("img")
                autonomous_img_tag["src"] = critical_item["croppedBase64"]
                autonomous_img_tag["alt"] = f"Recuperado: {critical_item['label']}"
                autonomous_img_tag["style"] = "max-width: 100%; margin: 16px 0; display: block;"
                
                if critical_item["label"] == "Logo":
                    page_soup_fragment.insert(0, autonomous_img_tag)
                else:
                    page_soup_fragment.append(autonomous_img_tag)

            # Unir el HTML de la página al HTML final
            if page_soup_fragment.body:
                for child_node in list(page_soup_fragment.body.children):
                    master_document_body.append(child_node)
            else:
                for child_node in list(page_soup_fragment.children):
                    master_document_body.append(child_node)

            # Pausa de cortesía para evitar saturar la API de Gemini
            if page_index < len(document_pages) - 1:
                aesthetic_page_divider = master_html_soup.new_tag("hr")
                aesthetic_page_divider["style"] = "border-top: 2px dashed #e2e8f0; margin: 40px 0;"
                master_document_body.append(aesthetic_page_divider)
                time.sleep(3.0)

        return str(master_html_soup)

    except Exception as pipeline_error:
        logging.error(f"Error en el pipeline: {pipeline_error}")
        raise HTTPException(status_code=500, detail=str(pipeline_error))
        
    finally:
        if os.path.exists(temporary_storage_path):
            os.remove(temporary_storage_path)