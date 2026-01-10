variable "aws_region" {
  description = "The AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_id" {
  description = "A unique identifier for naming resources (e.g. roles, streams)"
  type        = string
  default     = "predictive-maintenance"
}

variable "artifacts_bucket_name" {
  description = "The EXACT name of your existing S3 bucket"
  type        = string
  default     = "predictive-maintenance-artifacts-victor-obi"
}

variable "input_stream_name" {
  description = "Name of the Kinesis stream for raw data"
  type        = string
  default     = "predictive-maintenance-stream"
}

variable "output_stream_name" {
  description = "Name of the Kinesis stream for model predictions"
  type        = string
  default     = "predictive-maintenance-predictions"
}

variable "alert_email" {
  description = "Email address to receive Drift Alerts"
  type        = string
  default     = "obiezuechidera@gmail.com"
}

variable "firehose_role_name" {
  description = ""
  type        = string
  default     = "predictive-maintenance-firehose-role-tf" 
  }

variable "mlflow_uri" {
  type = string
}
variable "mlflow_username" {
  type = string
}
variable "mlflow_password" {
  type = string
  sensitive = true # Hides it from logs
}