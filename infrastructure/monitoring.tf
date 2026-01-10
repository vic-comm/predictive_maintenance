resource "aws_sns_topic" "drift_alerts" {
  name = "model-drift-alerts"
}

resource "aws_sns_topic_subscription" "email_target" {
  topic_arn = aws_sns_topic.drift_alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Triggers if 'DriftScore' is greater than 0.3 for 1 period (1 day)
resource "aws_cloudwatch_metric_alarm" "drift_alarm" {
  alarm_name          = "Production-Data-Drift"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "DriftScore"
  namespace           = "PredictiveMaintenance" # We will use this namespace in Python
  period              = "86400" # 24 Hours (checked once a day)
  statistic           = "Maximum"
  threshold           = "0.3"
  alarm_description   = "This metric monitors data drift in the production model."
  alarm_actions       = [aws_sns_topic.drift_alerts.arn]
}

# If you ever want your Prediction Lambda to push metrics directly,
# it needs permission to talk to CloudWatch.
resource "aws_iam_role_policy" "lambda_cloudwatch_metrics" {
  name = "lambda-cloudwatch-metrics-policy"
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "cloudwatch:PutMetricData",
          "sns:Publish"
        ],
        Resource = "*"
      }
    ]
  })
}