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
  # IMPORTANT: This must match your existing bucket name exactly!
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