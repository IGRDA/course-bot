terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Remote state in GCS. Configure bucket and prefix via -backend-config:
  #   terraform init -backend-config=backends/dev.conf
  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
}
