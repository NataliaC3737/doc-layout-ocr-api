FROM python:3.10-slim

WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el código (incluyendo la carpeta app/)
COPY . .

EXPOSE 8000

# Apunta al módulo app.main (dentro de la carpeta app/)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]