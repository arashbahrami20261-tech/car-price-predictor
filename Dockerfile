FROM python:3.11-slim

WORKDIR /app

# Dependencies are installed before the source is copied, so Docker can reuse
# the cached layer whenever only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ ./data/
COPY src/ ./src/
COPY tests/ ./tests/

# The model is trained at build time so the image ships ready to serve.
RUN python data/generate_data.py && python src/train.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
