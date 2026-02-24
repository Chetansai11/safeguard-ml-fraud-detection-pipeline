output "bronze_bucket_name" {
  description = "S3 bucket for raw (Bronze) data"
  value       = aws_s3_bucket.bronze.id
}

output "silver_bucket_name" {
  description = "S3 bucket for cleaned (Silver) data"
  value       = aws_s3_bucket.silver.id
}

output "gold_bucket_name" {
  description = "S3 bucket for ML-ready (Gold) data and artifacts"
  value       = aws_s3_bucket.gold.id
}

output "ecr_repository_url" {
  description = "ECR repository URL for inference container"
  value       = aws_ecr_repository.inference.repository_url
}

output "sagemaker_role_arn" {
  description = "IAM role ARN for SageMaker execution"
  value       = aws_iam_role.sagemaker.arn
}

output "mlflow_tracking_server_url" {
  description = "MLflow tracking server URL"
  value       = aws_sagemaker_mlflow_tracking_server.this.tracking_server_url
}
