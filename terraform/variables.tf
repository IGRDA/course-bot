variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod). Used to namespace all resources."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod"
  }
}

variable "region" {
  description = "GCP region for Cloud Run services and jobs"
  type        = string
  default     = "europe-west1"
}

variable "image" {
  description = "Container image for the Course Bot (e.g. gcr.io/my-project/course-bot:latest). Used for both the API server and the worker job."
  type        = string
}

variable "max_instances" {
  description = "Max instances for the API server Cloud Run service."
  type        = number
  default     = 3
}

variable "max_concurrent_jobs" {
  description = "Maximum number of concurrent Cloud Run Job executions. Enforced by the API server before dispatching."
  type        = number
  default     = 5
}

variable "job_timeout" {
  description = "Timeout for Cloud Run Job executions (worker). Must accommodate the longest expected Claude workflow."
  type        = string
  default     = "5400s"
}

variable "repo_url" {
  description = "Git repository URL for per-thread workspace isolation. Each Slack thread gets its own clone. Leave empty to disable."
  type        = string
  default     = ""
}

variable "default_branch" {
  description = "Default git branch for workspace isolation"
  type        = string
  default     = "main"
}
