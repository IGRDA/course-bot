#!/usr/bin/env bash
#
# Deploy course-bot to a target environment.
#
# Usage:
#   ./scripts/deploy.sh <environment>        # plan + apply infra only
#   ./scripts/deploy.sh <environment> --build # build & push image, then plan + apply
#
# Examples:
#   ./scripts/deploy.sh dev
#   ./scripts/deploy.sh dev --build
#   ./scripts/deploy.sh prod --build
#
set -euo pipefail

# -- Helpers ----------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

usage() {
  echo "Usage: $0 <environment> [--build]"
  echo ""
  echo "  environment   One of: dev, staging, prod"
  echo "  --build       Build and push Docker image before deploying infra"
  exit 1
}

# -- Parse arguments --------------------------------------------------------
ENV="${1:-}"
BUILD=false

if [[ -z "$ENV" ]]; then
  error "Missing environment argument."
  usage
fi

if [[ ! "$ENV" =~ ^(dev|staging|prod)$ ]]; then
  error "Invalid environment: $ENV. Must be one of: dev, staging, prod"
  exit 1
fi

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD=true; shift ;;
    *)       error "Unknown argument: $1"; usage ;;
  esac
done

# -- Resolve paths ----------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TF_DIR="${REPO_ROOT}/terraform"
VAR_FILE="${TF_DIR}/env/${ENV}.tfvars"
BACKEND_CONFIG="${TF_DIR}/backends/${ENV}.conf"

if [[ ! -f "$VAR_FILE" ]]; then
  error "Variable file not found: $VAR_FILE"
  exit 1
fi

if [[ ! -f "$BACKEND_CONFIG" ]]; then
  error "Backend config not found: $BACKEND_CONFIG"
  exit 1
fi

# Read project_id from the tfvars file for image tagging
PROJECT_ID=$(grep '^project_id' "$VAR_FILE" | sed 's/.*=[[:space:]]*"\(.*\)"/\1/')

info "Deploying to environment: ${ENV}"
info "GCP project:              ${PROJECT_ID}"

# -- Build & push image (optional) -----------------------------------------
if [[ "$BUILD" == true ]]; then
  # Use git short SHA for dev/staging, expect explicit tags for prod
  if [[ "$ENV" == "prod" ]]; then
    TAG="${IMAGE_TAG:-$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || echo "latest")}"
  else
    TAG="${IMAGE_TAG:-$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "latest")}"
  fi

  info "Image tag: ${TAG}"

  info "Building course-bot..."
  docker build --platform linux/amd64 -t "gcr.io/${PROJECT_ID}/course-bot:${TAG}" "${REPO_ROOT}"

  info "Pushing image..."
  docker push "gcr.io/${PROJECT_ID}/course-bot:${TAG}"

  # Update the tfvars in-place so Terraform picks up the new tag
  info "Updating image tag in ${VAR_FILE}..."
  sed -i.bak "s|image.*=.*|image       = \"gcr.io/${PROJECT_ID}/course-bot:${TAG}\"|" "$VAR_FILE"
  rm -f "${VAR_FILE}.bak"
fi

# -- Terraform --------------------------------------------------------------
# Use gcloud access token for Terraform auth if ADC is not configured
if [[ -z "${GOOGLE_OAUTH_ACCESS_TOKEN:-}" ]]; then
  export GOOGLE_OAUTH_ACCESS_TOKEN
  GOOGLE_OAUTH_ACCESS_TOKEN="$(gcloud auth print-access-token 2>/dev/null || true)"
fi

info "Initializing Terraform (backend: ${ENV})..."
terraform -chdir="$TF_DIR" init -backend-config="$BACKEND_CONFIG" -reconfigure

info "Planning..."
terraform -chdir="$TF_DIR" plan -var-file="$VAR_FILE" -out="tfplan-${ENV}"

echo ""
info "Review the plan above. Apply?"
read -rp "  Type 'yes' to apply: " CONFIRM

if [[ "$CONFIRM" == "yes" ]]; then
  info "Applying..."
  terraform -chdir="$TF_DIR" apply "tfplan-${ENV}"

  echo ""
  info "Deploy complete! Outputs:"
  terraform -chdir="$TF_DIR" output
else
  warn "Apply cancelled."
fi

# Clean up plan file
rm -f "${TF_DIR}/tfplan-${ENV}"
