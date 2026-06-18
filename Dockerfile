FROM python:3.10-slim

# 1. Instalar dependencias del sistema operativo
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# 3. Copiar e instalar los requerimientos de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar la estructura de código manteniendo la carpeta /app
COPY ./app ./app

# También copiamos main.py si está en la raíz de tu proyecto local
COPY main.py . 

# 5. Descargar los pesos del modelo durante la construcción
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='anyformat/doclayout-yolo-docstructbench', filename='model.pt', local_dir='.', local_dir_use_symlinks=False)" && \
    mv model.pt doclayout_yolo_core.pt

# 6. Exponer el puerto estándar que utiliza Google Cloud Run
EXPOSE 8080

# 7. Comando de arranque de Uvicorn (apuntando correctamente a main:app en la raíz)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]