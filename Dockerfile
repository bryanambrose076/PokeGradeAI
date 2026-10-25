FROM python:3.13-slim

# Lambda Web Adapter: on Lambda it turns invocations into HTTP requests to uvicorn; elsewhere it does nothing
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8000
ENV AWS_LWA_READINESS_CHECK_PATH=/health

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# keep the model inside /app: Lambda doesn't run as root, so it can't read /root/.cache
ENV HF_HOME=/app/.hf_cache
RUN python -c "from transformers import AutoModel, AutoImageProcessor; AutoModel.from_pretrained('facebook/dinov2-base'); AutoImageProcessor.from_pretrained('facebook/dinov2-base')"
ENV HF_HUB_OFFLINE=1

COPY . .

CMD ["uvicorn", "pokegrade.main:app", "--host", "0.0.0.0", "--port", "8000"]
