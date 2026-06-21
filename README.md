# Document layout & reconstruction API

Este repositorio contiene el backend y microservicio desarrollado en **FastAPI** para la digitalización inteligente de documentos. 

El sistema utiliza un pipeline híbrido que combina inteligencia artificial local y en la nube: **DocLayout-YOLOv10** se encarga de la segmentación visual y espacial del documento (detección de tablas, imágenes, firmas, logos y sellos), mientras que **Google Gemini** realiza un OCR estructural y semántico para empaquetar todo en un documento **HTML5 limpio, accesible y fiel al diseño original**.

---

## Arquitectura del pipeline

1. **Rasterización:** Convierte archivos PDF o imágenes (RGBA/LA) a un espacio de color normalizado RGB usando `pdf2image` y `Pillow`.
2. **Segmentación local (YOLO):** Detecta bounding boxes y extrae las regiones de interés no textuales (Logos, Firmas, Sellos, Tablas).
3. **Estructuración multimodal (Gemini):** Transcribe el texto del documento directamente a etiquetas semánticas HTML (`<h1>`, `<p>`, `<ul>`, `<table>`).
4. **Reconstrucción híbrida:** Fusiona las respuestas del LLM con los recortes en formato *Data URL (Base64)* procesados por OpenCV, asegurando que ningún componente crítico (como firmas o sellos) se pierda.

---

## Requisitos previos y configuración

### 1. Requisitos del sistema
Asegúrate de tener instaladas las siguientes herramientas en tu sistema operativo:
* **Python 3.10 o superior**
* **Poppler** (Obligatorio para que `pdf2image` pueda rasterizar PDFs).
  * *Ubuntu/Debian:* `sudo apt-get install poppler-utils`
  * *MacOS (Homebrew):* `brew install poppler`
  * *Windows:* Descargar los binarios de Poppler y añadirlos al `PATH` del sistema.

### 2. Descarga del modelo local (Obligatorio)

Por motivos de peso, los archivos de entrenamiento del modelo de visión artificial no están incluidos en este repositorio. Es necesario descargar los pesos de forma externa:

1. **Descargar el modelo:** Haz clic en el siguiente enlace para acceder al repositorio oficial en Hugging Face:
   [**anyformat/doclayout-yolo-docstructbench**](https://huggingface.co/anyformat/doclayout-yolo-docstructbench)
2. **Seleccionar el archivo:** Busca y descarga específicamente el archivo llamado **`doclayout_yolo_core.pt`**.
3. **Ubicación:** Coloca el archivo descargado directamente en la **raíz de este proyecto**.

>  **Nota:** El archivo `.env` por defecto buscará este nombre en la raíz (`MODELO_YOLO_PATH=doclayout_yolo_core.pt`). Si decides cambiarlo de carpeta, asegúrate de actualizar la ruta en tus variables de entorno.

### 3. Variables de entorno
Crea un archivo `.env` en la raíz del proyecto basándote en la siguiente configuración:

```env
MODELO_YOLO_PATH=doclayout_yolo_core.pt
GEMINI_API_KEY=tu_google_gemini_api_key_aqui
