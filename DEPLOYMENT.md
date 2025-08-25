# AWS Deployment Guide for Medical Summarizer API

This guide provides step-by-step instructions for deploying the Medical Summarizer API on AWS using containerized microservices with real-time API support.

## 🏗️ Architecture Overview

The deployment uses a modern microservices architecture with the following components:

- **ECS Fargate**: Container orchestration for the API service
- **Application Load Balancer**: Traffic distribution and SSL termination
- **RDS PostgreSQL**: Persistent data storage
- **ElastiCache Redis**: Caching and message queuing
- **ECR**: Container image registry
- **CloudWatch**: Monitoring and logging
- **VPC**: Network isolation and security

## 📋 Prerequisites

### Required Tools
- [AWS CLI](https://aws.amazon.com/cli/) (v2.x)
- [Docker](https://www.docker.com/) (v20.x+)
- [Terraform](https://www.terraform.io/) (v1.0+)
- [Git](https://git-scm.com/)

### AWS Requirements
- AWS Account with appropriate permissions
- IAM user with programmatic access
- Access to ECS, ECR, RDS, ElastiCache, and VPC services

## 🚀 Quick Start Deployment

### 1. Clone and Setup Repository

```bash
git clone <your-repo-url>
cd med-summarizer-api
```

### 2. Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and default region
```

### 3. Set Environment Variables

```bash
export AWS_REGION=us-east-1
export POSTGRES_PASSWORD=$(openssl rand -base64 32)
export PROJECT_NAME=med-summarizer
```

### 4. Run Deployment Script

```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

## 🔧 Manual Deployment Steps

### Step 1: Build and Push Docker Image

```bash
# Build production image
docker build -f Dockerfile.prod -t med-summarizer:latest .

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-api"

# Create ECR repository
aws ecr create-repository --repository-name ${PROJECT_NAME}-api --region ${AWS_REGION}

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO_URI}

# Tag and push image
docker tag med-summarizer:latest ${ECR_REPO_URI}:latest
docker push ${ECR_REPO_URI}:latest
```

### Step 2: Deploy Infrastructure with Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="postgres_password=${POSTGRES_PASSWORD}"

# Apply changes
terraform apply -auto-approve -var="postgres_password=${POSTGRES_PASSWORD}"

# Get outputs
terraform output
```

### Step 3: Verify Deployment

```bash
# Get ALB DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://${ALB_DNS}/health

# Test API documentation
curl http://${ALB_DNS}/docs
```

## 🔐 Security Configuration

### SSL/TLS Setup

1. **Generate SSL Certificate** (for production):
```bash
# Using Let's Encrypt
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates to nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/key.pem
```

2. **Update nginx configuration** with your domain name

### IAM Roles and Policies

The Terraform configuration creates the necessary IAM roles:
- `ecs-execution-role`: For ECS task execution
- `ecs-task-role`: For ECS task runtime permissions

## 📊 Monitoring and Observability

### CloudWatch Metrics
- ECS service metrics (CPU, memory, network)
- RDS performance insights
- ALB access logs
- Custom application metrics

### Prometheus + Grafana
- Application metrics collection
- Custom dashboards
- Alerting rules

### Log Aggregation
- ECS task logs in CloudWatch
- Application logs with structured logging
- Centralized log analysis

## 🔄 CI/CD Pipeline

### GitHub Actions Setup

1. **Add Repository Secrets**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `POSTGRES_PASSWORD`

2. **Push to Main Branch**: Automatically triggers deployment

3. **Manual Deployment**: Use GitHub Actions workflow dispatch

### Deployment Pipeline Stages

1. **Test**: Run unit tests and linting
2. **Build**: Build and push Docker image to ECR
3. **Deploy**: Apply Terraform infrastructure changes
4. **Verify**: Health checks and smoke tests

## 🚨 Troubleshooting

### Common Issues

#### ECS Service Not Starting
```bash
# Check ECS service events
aws ecs describe-services --cluster med-summarizer-cluster --services med-summarizer-api-service

# Check task logs
aws logs describe-log-groups --log-group-name-prefix "/ecs/med-summarizer-api"
```

#### Database Connection Issues
```bash
# Verify RDS security group allows ECS tasks
aws ec2 describe-security-groups --group-ids <rds-security-group-id>

# Check RDS endpoint accessibility
telnet <rds-endpoint> 5432
```

#### Load Balancer Health Check Failures
```bash
# Check target group health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>

# Verify container health check endpoint
curl -f http://localhost:8000/health
```

### Log Analysis

```bash
# View application logs
aws logs tail /ecs/med-summarizer-api --follow

# View nginx logs
docker logs <nginx-container-id>
```

## 📈 Scaling and Performance

### Auto Scaling

```bash
# Update ECS service desired count
aws ecs update-service --cluster med-summarizer-cluster --service med-summarizer-api-service --desired-count 5
```

### Performance Tuning

1. **Container Resources**: Adjust CPU/memory in `terraform/variables.tf`
2. **Database**: Scale RDS instance class and storage
3. **Caching**: Optimize Redis configuration
4. **Load Balancer**: Configure connection draining and health check intervals

## 🧹 Cleanup

### Remove All Resources

```bash
cd terraform
terraform destroy -auto-approve -var="postgres_password=${POSTGRES_PASSWORD}"
```

### Remove ECR Repository

```bash
aws ecr delete-repository --repository-name med-summarizer-api --force --region ${AWS_REGION}
```

## 📚 Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🆘 Support

For deployment issues:
1. Check CloudWatch logs for error details
2. Verify AWS service quotas and limits
3. Review security group and IAM permissions
4. Check Terraform state and plan output

---

**Note**: This deployment is designed for production use but should be customized based on your specific requirements, compliance needs, and security policies. 