# ---------- API Server (Cloud Run Service) ----------
#
# Lightweight HTTP server that receives Slack events, runs filters and
# acknowledgers, and dispatches heavy work to Cloud Run Job executions.
# Lightweight events (reactions, member_joined) are handled in-process.

resource "google_cloud_run_v2_service" "course_bot" {
  name     = "course-bot-${var.environment}"
  location = var.region

  depends_on = [
    google_secret_manager_secret_iam_member.signing_secret_access,
    google_secret_manager_secret_iam_member.bot_token_access,
  ]

  template {
    service_account = google_service_account.course_bot.email
    timeout         = "30s"

    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    max_instance_request_concurrency = 80

    containers {
      image = var.image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      env {
        name = "SLACK_SIGNING_SECRET"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.slack_signing_secret.secret_id
            version = "latest"
          }
        }
      }

      env {
        name = "SLACK_BOT_TOKEN"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.slack_bot_token.secret_id
            version = "latest"
          }
        }
      }

      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "GCP_REGION"
        value = var.region
      }

      env {
        name  = "JOB_NAME"
        value = google_cloud_run_v2_job.course_bot_worker.name
      }

      env {
        name  = "MAX_CONCURRENT_JOBS"
        value = tostring(var.max_concurrent_jobs)
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 10
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }
  }
}

# ---------- Worker (Cloud Run Job) ----------
#
# Runs the heavy processing: workspace setup, Claude agent invocation,
# Slack response.  Each execution handles a single Slack event.

resource "google_cloud_run_v2_job" "course_bot_worker" {
  name     = "course-bot-worker-${var.environment}"
  location = var.region

  depends_on = [
    google_secret_manager_secret_iam_member.bot_token_access,
    google_secret_manager_secret_iam_member.claude_token_access,
    google_secret_manager_secret_iam_member.mistral_key_access,
    google_secret_manager_secret_iam_member.google_books_key_access,
    google_secret_manager_secret_iam_member.youtube_key_access,
  ]

  template {
    task_count  = 1
    parallelism = 1

    template {
      service_account  = google_service_account.course_bot.email
      timeout          = var.job_timeout
      max_retries      = 0

      containers {
        image   = var.image
        command = ["python", "-m", "bot.job.worker"]

        resources {
          limits = {
            cpu    = "8"
            memory = "16Gi"
          }
        }

        env {
          name = "SLACK_BOT_TOKEN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.slack_bot_token.secret_id
              version = "latest"
            }
          }
        }

        env {
          name = "CLAUDE_CODE_OAUTH_TOKEN"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.claude_code_oauth_token.secret_id
              version = "latest"
            }
          }
        }

        env {
          name = "MISTRAL_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.mistral_api_key.secret_id
              version = "latest"
            }
          }
        }

        env {
          name = "GOOGLE_BOOKS_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.google_books_api_key.secret_id
              version = "latest"
            }
          }
        }

        env {
          name = "YOUTUBE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.youtube_api_key.secret_id
              version = "latest"
            }
          }
        }

        env {
          name  = "CLAUDE_AGENT_DIR"
          value = "/app/engine"
        }

        env {
          name  = "MISTRAL_MODEL_NAME"
          value = "mistral-medium-latest"
        }

        env {
          name  = "REPO_URL"
          value = var.repo_url
        }

        env {
          name  = "DEFAULT_BRANCH"
          value = var.default_branch
        }

        env {
          name  = "CLAUDE_RESPONSE_TIMEOUT"
          value = trimsuffix(var.job_timeout, "s")
        }
      }
    }
  }
}
