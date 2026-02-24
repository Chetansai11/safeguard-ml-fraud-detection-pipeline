# =============================================================================
# IAM Role — Least-Privileged SageMaker Execution Role
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

resource "aws_iam_role" "sagemaker" {
  name = "${local.name_prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = [
            "sagemaker.amazonaws.com",
            "lambda.amazonaws.com",
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# --- S3 Access: Scoped to project buckets only ---
resource "aws_iam_role_policy" "s3_access" {
  name = "${local.name_prefix}-s3-access"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListProjectBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
        ]
        Resource = [
          aws_s3_bucket.bronze.arn,
          aws_s3_bucket.silver.arn,
          aws_s3_bucket.gold.arn,
        ]
      },
      {
        Sid    = "ReadWriteProjectObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
        ]
        Resource = [
          "${aws_s3_bucket.bronze.arn}/*",
          "${aws_s3_bucket.silver.arn}/*",
          "${aws_s3_bucket.gold.arn}/*",
        ]
      }
    ]
  })
}

# --- SageMaker: Pipeline, Training, Processing, Model Registry ---
resource "aws_iam_role_policy" "sagemaker_access" {
  name = "${local.name_prefix}-sagemaker-access"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SageMakerPipeline"
        Effect = "Allow"
        Action = [
          "sagemaker:CreatePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutionSteps",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.project_name}*"
      },
      {
        Sid    = "SageMakerTrainingProcessing"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:*/*"
      },
      {
        Sid    = "SageMakerModelRegistry"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModelPackage",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModelPackage",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackages",
          "sagemaker:UpdateModelPackage",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model-package*/${var.project_name}*"
      },
      {
        Sid    = "PassRole"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/${local.name_prefix}-sagemaker-role"
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

# --- ECR: Pull images for training/inference containers ---
resource "aws_iam_role_policy" "ecr_access" {
  name = "${local.name_prefix}-ecr-access"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "ECRPullPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = aws_ecr_repository.inference.arn
      }
    ]
  })
}

# --- CloudWatch Logs: Training and processing job logs ---
resource "aws_iam_role_policy" "cloudwatch_logs" {
  name = "${local.name_prefix}-cw-logs"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
        ]
        Resource = "arn:${data.aws_partition.current.partition}:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
      }
    ]
  })
}
