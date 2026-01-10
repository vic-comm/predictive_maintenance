provider "aws" {
  region = "us-east-1"
}

# --- 1. THE INPUT STREAM ("The Mail Slot") ---
resource "aws_kinesis_stream" "input_stream" {
  name             = var.input_stream_name
  shard_count      = 1
  retention_period = 24
}

# --- 2. THE OUTPUT STREAM ("The Outbox") ---
resource "aws_kinesis_stream" "output_stream" {
  name             = var.output_stream_name
  shard_count      = 1
  retention_period = 24
}

data "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket_name
}

# --- 4. FIREHOSE ("The Delivery Truck") ---
resource "aws_iam_role" "firehose_role" {
  name = var.firehose_role_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "firehose.amazonaws.com" }}]
  })
}

# Firehose Policy (Read from Stream, Write to S3)
resource "aws_iam_role_policy" "firehose_policy" {
  name = "firehose-policy"
  role = aws_iam_role.firehose_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow",
        Action = ["s3:PutObject", "s3:GetBucketLocation"],
        Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"]
      },
      {
        Effect = "Allow",
        Action = ["kinesis:GetRecords", "kinesis:DescribeStream"],
        Resource = aws_kinesis_stream.output_stream.arn
      }
    ]
  })
}

resource "aws_kinesis_firehose_delivery_stream" "s3_saver" {
  name        = "KDS-S3-z5TgK-tf"
  destination = "s3"

  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.output_stream.arn
    role_arn           = aws_iam_role.firehose_role.arn
  }

  s3_configuration {
    role_arn   = aws_iam_role.firehose_role.arn
    bucket_arn = aws_s3_bucket.artifacts.arn
    prefix     = "predictions/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"
    buffer_interval = 60
  }
}

# --- 5. LAMBDA ("The Worker") ---
# First, create the ECR Repo to store the Docker Image
resource "aws_ecr_repository" "lambda_repo" {
  name = "predictive-maintenance-lambda"
}

# Lambda IAM Role
resource "aws_iam_role" "lambda_role" {
  name = "predictive-maintenance-lambda-role-tf"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "lambda.amazonaws.com" }}]
  })
}

# Lambda Policy (Read Input, Write Output, Logging)
resource "aws_iam_role_policy" "lambda_policy" {
  name = "lambda-policy"
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow",
        Action = ["kinesis:GetRecords", "kinesis:GetShardIterator", "kinesis:DescribeStream", "kinesis:ListStreams"],
        Resource = aws_kinesis_stream.input_stream.arn
      },
      {
        Effect = "Allow",
        Action = "kinesis:PutRecord",
        Resource = aws_kinesis_stream.output_stream.arn
      },
      {
        Effect = "Allow",
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow",
        Action = "s3:GetObject",
        Resource = "arn:aws:s3:::*" # Needed to load MLflow model
      }
    ]
  })
}

# The Function Itself
resource "aws_lambda_function" "prediction_function" {
  function_name = "predictive-maintenance-function-tf"
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"
  # Placeholder image until you push the real one
  image_uri     = "${aws_ecr_repository.lambda_repo.repository_url}:latest" 
  timeout       = 60
  memory_size   = 1024
}

# The Trigger (Connect Input Stream to Lambda)
resource "aws_lambda_event_source_mapping" "kinesis_trigger" {
  event_source_arn  = aws_kinesis_stream.input_stream.arn
  function_name     = aws_lambda_function.prediction_function.arn
  starting_position = "LATEST"
  batch_size        = 10
}
