FROM python:3.12-slim-bookworm

WORKDIR /app

# Instalar dependencias del sistema si luego las necesitas (ej. geospatial). Por ahora, base mínima.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código y datos
COPY . .

# Puerto estándar de uvicorn
ENV PORT=8000
EXPOSE 8000

# Arranque
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
