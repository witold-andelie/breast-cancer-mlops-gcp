#!/usr/bin/env bash
set -euo pipefail

# Parameterized GCP Artifact Registry + Cloud Run deployment script for cancer_mlops.
# Usage:
#   PROJECT_ID=my-gcp-project REGION=europe-west3 ./cloud/gcp/deploy_cloud_run.sh

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your GCP project ID}"
REGION="${REGION:-europe-west3}"
REPO="${REPO:-mlops-repo}"
SERVICE="${SERVICE:-cancer-mlops-api}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"
GCS_DATA_URI="${GCS_DATA_URI:-}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${IMAGE_TAG}"

echo "==> Ensuring Artifact Registry repository '${REPO}' exists in ${REGION}..."
gcloud artifacts repositories describe "${REPO}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Docker images for Cancer MLOps services"

echo "==> Building and pushing container image via Cloud Build: ${IMAGE_URI}"
gcloud builds submit \
  --project="${PROJECT_ID}" \
  --tag="${IMAGE_URI}" \
  .

ENV_VARS="PYTHONUNBUFFERED=1"
if [[ -n "${GCS_DATA_URI}" ]]; then
  ENV_VARS="${ENV_VARS},CANCER_MLOPS_DATA_URI=${GCS_DATA_URI}"
fi

echo "==> Deploying ${SERVICE} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --image="${IMAGE_URI}" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --cpu=1 \
  --memory=1Gi \
  --min-instances=0 \
  --max-instances=3 \
  --set-env-vars="${ENV_VARS}"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo "==> Deployed URL: ${SERVICE_URL}"
echo "==> Verifying /health endpoint..."
curl -fsS "${SERVICE_URL}/health"
echo ""
