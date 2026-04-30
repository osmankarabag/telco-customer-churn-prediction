FROM python:3.12-slim

WORKDIR /app

# Sistem bagimliliklari
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Python bagimliliklari
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalari
COPY . .

# Port
EXPOSE 8000

# Saglik kontrolu
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Calistirma
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
