# Service account for Course Bot (shared by API server and worker job)
resource "google_service_account" "course_bot" {
  account_id   = "course-bot-${var.environment}"
  display_name = "Course Bot (${var.environment})"
  description  = "Service account for the Course Bot Cloud Run service and worker job (${var.environment})"
}

# Allow unauthenticated access to Course Bot (Slack sends events to public URL)
resource "google_cloud_run_v2_service_iam_member" "course_bot_public" {
  location = google_cloud_run_v2_service.course_bot.location
  name     = google_cloud_run_v2_service.course_bot.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Allow the service account to run job executions with env overrides
# (run.jobs.runWithOverrides requires roles/run.developer, not just run.invoker)
resource "google_cloud_run_v2_job_iam_member" "course_bot_job_invoker" {
  location = google_cloud_run_v2_job.course_bot_worker.location
  name     = google_cloud_run_v2_job.course_bot_worker.name
  role     = "roles/run.developer"
  member   = "serviceAccount:${google_service_account.course_bot.email}"
}

# Allow the service account to list job executions (capacity/thread-busy checks)
resource "google_project_iam_member" "course_bot_run_viewer" {
  project = var.project_id
  role    = "roles/run.viewer"
  member  = "serviceAccount:${google_service_account.course_bot.email}"
}

# ---------- Secret Manager ----------
# Secret containers (values are managed via gcloud, not Terraform)

resource "google_secret_manager_secret" "slack_signing_secret" {
  secret_id = "slack-signing-secret"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "slack_bot_token" {
  secret_id = "slack-bot-token"

  replication {
    auto {}
  }
}

# Grant the Cloud Run service account access to read the secrets
resource "google_secret_manager_secret_iam_member" "signing_secret_access" {
  secret_id = google_secret_manager_secret.slack_signing_secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}

resource "google_secret_manager_secret_iam_member" "bot_token_access" {
  secret_id = google_secret_manager_secret.slack_bot_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}

resource "google_secret_manager_secret" "claude_code_oauth_token" {
  secret_id = "claude-code-oauth-token"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "claude_token_access" {
  secret_id = google_secret_manager_secret.claude_code_oauth_token.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}

resource "google_secret_manager_secret" "mistral_api_key" {
  secret_id = "mistral-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "mistral_key_access" {
  secret_id = google_secret_manager_secret.mistral_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}

resource "google_secret_manager_secret" "google_books_api_key" {
  secret_id = "google-books-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "google_books_key_access" {
  secret_id = google_secret_manager_secret.google_books_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}

resource "google_secret_manager_secret" "youtube_api_key" {
  secret_id = "youtube-api-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_iam_member" "youtube_key_access" {
  secret_id = google_secret_manager_secret.youtube_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.course_bot.email}"
}
