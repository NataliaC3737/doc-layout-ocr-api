# -*- coding: utf-8 -*-
import os
import re
import logging
from collections import Counter
from PIL import Image
from pdf2image import convert_from_path
from bs4 import BeautifulSoup

def rasterize_document_to_rgb_images(file_path: str) -> list[Image.Image]:
    """
    Detecta el tipo de archivo (PDF o Imagen estándar) y lo convierte en una 
    lista de objetos PIL.Image normalizados en el espacio de color RGB.
    Elimina preventivamente canales de transparencia para evitar fallos de compresión.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        logging.info("Archivo PDF detectado. Iniciando rasterización con pdf2image a 150 DPI...")
        document_pages = convert_from_path(file_path, dpi=150)
        logging.info(f"Se han extraído exitosamente {len(document_pages)} páginas del PDF.")
        return document_pages
    else:
        logging.info("Archivo de imagen estándar detectado. Cargando en memoria...")
        source_image = Image.open(file_path)
        
        # Normalización de imágenes con canal alfa (RGBA / LA / Paleta con transparencia)
        if source_image.mode in ('RGBA', 'LA') or (source_image.mode == 'P' and 'transparency' in source_image.info):
            logging.info(f"Detectado canal de transparencia ({source_image.mode}). Normalizando a RGB con fondo blanco...")
            rgb_background = Image.new("RGB", source_image.size, (255, 255, 255))
            rgb_background.paste(source_image, mask=source_image.split()[3] if source_image.mode == 'RGBA' else None)
            return [rgb_background]
            
        return [source_image]
