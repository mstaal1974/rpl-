FROM python:3.12-slim
# ffmpeg: decode/normalise competency-conversation recordings (audio + video) for transcription
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY units/ ./units/
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app
ENV PORT=8080
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
