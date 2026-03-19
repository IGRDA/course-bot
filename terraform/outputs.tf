output "environment" {
  description = "Current environment name"
  value       = var.environment
}

output "course_bot_url" {
  description = "Public URL of the Course Bot API server (use this in Slack Event Subscriptions)"
  value       = google_cloud_run_v2_service.course_bot.uri
}

output "course_bot_service_account" {
  description = "Email of the Course Bot service account"
  value       = google_service_account.course_bot.email
}

output "course_bot_job_name" {
  description = "Name of the Course Bot worker Cloud Run Job"
  value       = google_cloud_run_v2_job.course_bot_worker.name
}
