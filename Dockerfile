FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y socat && rm -rf /var/lib/apt/lists/*
COPY . .
ENV PYTHONPATH=/app
EXPOSE 7860 8000
CMD ["bash", "start.sh"]
