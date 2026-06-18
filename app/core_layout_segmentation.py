# -*- coding: utf-8 -*-
import os
from time import time, sleep
import base64
import logging
import numpy as np
import cv2
from PIL import Image
from doclayout_yolo import YOLOv10
from google import genai
from google.genai import types
from io import BytesIO
from pydantic import BaseModel, Field

# Esquema de Pydantic para forzar el Structured Output
class DocumentLayout(BaseModel):
    html: str = Field(description="Código HTML estructurado completo con los escapes necesarios según las pautas.")

class LocalLayoutVisionDetector:
    """
    Orquesta el modelo local de visión artificial (DocLayout-YOLOv10) para la 
    detección, localización espacial y extracción geométrica de regiones no textuales.
    """
    def __init__(self, trained_model_path="doclayout_yolo_core.pt"):
        if not os.path.exists(trained_model_path):
            raise FileNotFoundError(f"Pesos del modelo no localizados en la ruta: {trained_model_path}")

        logging.info(f"Inicializando detector de layout local con pesos: {trained_model_path}")

        self.vision_engine = YOLOv10(trained_model_path)

    def _evaluate_local_ink_density(self, cropped_roi: np.ndarray, box_width: int, box_height: int) -> float:
        """Calcula la densidad relativa de píxeles oscuros sobre una región segmentada."""
        if cropped_roi.size == 0 or box_width == 0 or box_height == 0:
            return 0.0

        grayscale_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)

        _, binarized_threshold = cv2.threshold(grayscale_roi, 240, 255, cv2.THRESH_BINARY_INV)

        return float(cv2.countNonZero(binarized_threshold) / (box_width * box_height))

    def segment_document_layout(self, pil_image: Image.Image) -> dict:
        """
        Analiza las propiedades espaciales del documento e indexa las regiones de interés (ROI)
        aplicando un mapeo semántico de precisión según su ubicación y densidad de tinta.
        """
        rgb_image = pil_image.convert('RGB')
        opencv_bgr_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        canvas_height, canvas_width, _ = opencv_bgr_image.shape

        # Inferencia optimizada a 1024px nativos para análisis documental
        inference_results = self.vision_engine(
            opencv_bgr_image, verbose=False, imgsz=1024, conf=0.25, iou=0.45, agnostic_nms=True
        )[0]

        extracted_regions = []

        for index, bounding_box in enumerate(inference_results.boxes):
            coordinates = bounding_box.xyxy[0].tolist()
            confidence_score = float(bounding_box.conf[0])
            class_id = int(bounding_box.cls[0])

            if confidence_score < 0.30:
                continue

            raw_model_label = inference_results.names[class_id].lower().strip()
            x1, y1, x2, y2 = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
            box_width, box_height = x2 - x1, y2 - y1

            cropped_roi = opencv_bgr_image[y1:y2, x1:x2]
            if cropped_roi.size == 0:
                continue

            ink_density = self._evaluate_local_ink_density(cropped_roi, box_width, box_height)
            semantic_label = None

            # Reglas de negocio del mapeo semántico híbrido
            if 'table' in raw_model_label:
                semantic_label = 'Tabla'
            elif 'figure' in raw_model_label or 'picture' in raw_model_label:
                if (y1 / canvas_height) < 0.25 and box_width < (canvas_width * 0.45):
                    semantic_label = 'Logo' if ink_density > 0.10 else 'Imagen'
                else:
                    semantic_label = 'Imagen'
            elif 'title' in raw_model_label:
                if (y1 / canvas_height) < 0.20 and box_width < (canvas_width * 0.40) and ink_density > 0.15:
                    semantic_label = 'Logo'
                else:
                    continue  # Delegado al análisis estructural del LLM
            elif 'abandon' in raw_model_label:
                if 0.02 < ink_density < 0.25:
                    semantic_label = 'Firma' if (y1 / canvas_height) > 0.50 else 'Sello'
                else:
                    continue
            else:
                continue

            # Codificación a formato Data URL (Base64)
            _, image_buffer = cv2.imencode('.png', cropped_roi)
            base64_encoded_string = base64.b64encode(image_buffer).decode('utf-8')
            data_url_payload = f"data:image/png;base64,{base64_encoded_string}"
            unique_region_id = f"region_{int(time())}_{index}"

            extracted_regions.append({
                "id": unique_region_id, "label": semantic_label, "confidence": confidence_score,
                "x": x1, "y": y1, "width": box_width, "height": box_height,
                "croppedBase64": data_url_payload, "density": ink_density
            })

        return {
            "originalWidth": canvas_width, 
            "originalHeight": canvas_height, 
            "regions": extracted_regions
        }


class MultimodalStructureExtractor:
    """
    Gestiona la comunicación con los modelos multimodales de la API de Google Gemini 
    para realizar OCR estructural directo a código HTML5 semántico y limpio.
    """
    def __init__(self, google_api_key: str):
        if not google_api_key:
            raise ValueError("Se requiere una API Key válida para instanciar el cliente de Google GenAI.")

        self.gemini_client = genai.Client(api_key=google_api_key)
        self.orchestrated_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.5-flash"]

    def convert_page_to_semantic_html(self, pil_image: Image.Image, page_index: int = 0) -> str:
        """Envía la imagen a Gemini aplicando estrategias robustas de reintentos y tolerancia a fallos."""
        memory_buffer = BytesIO()
        pil_image.save(memory_buffer, format="JPEG", quality=95)
        image_bytes_payload = memory_buffer.getvalue()

        system_instruction_prompt = f"""
            Eres un motor avanzado de análisis y estructuración de layouts de documentos. Tu objetivo es procesar las imágenes suministradas, detectar con precisión la disposición visual de todos sus elementos y compilar este layout de alta fidelidad en código HTML estructurado.
            Sigue rigurosamente estas pautas:
            1. Analiza el orden de los elementos tal como están ubicados en las páginas (de arriba a abajo).
            2. Traduce los bloques detectados a los siguientes tags estándar de HTML estructurado: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <table>.
            NORMAS CRÍTICAS DE LIMPIEZA DE CÓDIGO HTML:
            - Queda TOTALMENTE PROHIBIDO el uso de etiquetas <span> con estilos en línea o saltos de línea vacíos (<br>).
            3. DETECCIÓN E INCLUSIÓN DE IMÁGENES:
              - Si detectas cualquier imagen, gráfico, firma manuscrita, sello, dibujo o logotipo, inserta la etiqueta HTML exacta:
                <img src="ORIGINAL_IMAGE_{page_index}" alt="Componente detectado" />
        """
        
        request_config = types.GenerateContentConfig(
            response_mime_type="application/json", 
            response_schema=DocumentLayout,
            temperature=0.1
        )

        multimodal_image_part = types.Part.from_bytes(data=image_bytes_payload, mime_type="image/jpeg")
        last_encountered_exception = "No exceptions registered"

        for model_name in self.orchestrated_models:
            max_execution_attempts = 2

            for attempt in range(1, max_execution_attempts + 1):
                try:
                    api_response = self.gemini_client.models.generate_content(
                        model=model_name, 
                        contents=[multimodal_image_part, system_instruction_prompt], 
                        config=request_config
                    )

                    if api_response and api_response.text:
                        return api_response.text

                except Exception as api_exception:
                    last_encountered_exception = str(api_exception)

                    if ("503" in last_encountered_exception.lower() or "exhausted" in last_encountered_exception.lower()) and attempt < max_execution_attempts:
                        # Throttling adaptivo ante saturación de cuota (Rate limits)
                        sleep(attempt * 3.0)
                        
                    else:
                        break
                        
        raise RuntimeError(f"El pipeline multimodal falló en todos los modelos configurados. Detalle técnico: {last_encountered_exception}")