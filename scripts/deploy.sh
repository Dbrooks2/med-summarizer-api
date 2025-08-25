#!/bin/bash

# Medical Summarizer API Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="med-summarizer"
AWS_REGION=${AWS_REGION:-"us-east-1"}
ECR_REPO_NAME="${PROJECT_NAME}-api"
IMAGE_TAG=${IMAGE_TAG:-"latest"}

echo -e "${GREEN}🚀 Starting deployment of ${PROJECT_NAME}...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform is not installed. Please install it first.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Prerequisites check passed${NC}"

# Build Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -f Dockerfile.prod -t ${PROJECT_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

# Create ECR repository if it doesn't exist
echo -e "${YELLOW}🏗️  Creating ECR repository if it doesn't exist...${NC}"
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

# Login to ECR
echo -e "${YELLOW}🔐 Logging into ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPO_URI}

# Tag and push image
echo -e "${YELLOW}📤 Tagging and pushing image to ECR...${NC}"
docker tag ${PROJECT_NAME}:${IMAGE_TAG} ${ECR_REPO_URI}:${IMAGE_TAG}
docker push ${ECR_REPO_URI}:${IMAGE_TAG}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Image pushed to ECR successfully${NC}"
else
    echo -e "${RED}❌ Failed to push image to ECR${NC}"
    exit 1
fi

# Deploy infrastructure with Terraform
echo -e "${YELLOW}🏗️  Deploying infrastructure with Terraform...${NC}"
cd terraform

# Initialize Terraform
terraform init

# Plan deployment
echo -e "${YELLOW}📋 Planning Terraform deployment...${NC}"
terraform plan -var="postgres_password=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"

# Apply changes
echo -e "${YELLOW}🚀 Applying Terraform changes...${NC}"
terraform apply -auto-approve -var="postgres_password=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"

# Get outputs
echo -e "${YELLOW}📊 Getting deployment outputs...${NC}"
ALB_DNS=$(terraform output -raw alb_dns_name)
ECR_REPO_URL=$(terraform output -raw ecr_repository_url)

cd ..

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Application Load Balancer: http://${ALB_DNS}${NC}"
echo -e "${GREEN}📦 ECR Repository: ${ECR_REPO_URL}${NC}"
echo -e "${GREEN}📚 API Documentation: http://${ALB_DNS}/docs${NC}"
echo -e "${GREEN}💚 Health Check: http://${ALB_DNS}/health${NC}"

# Wait for service to be ready
echo -e "${YELLOW}⏳ Waiting for service to be ready...${NC}"
sleep 30

# Test the deployment
echo -e "${YELLOW}🧪 Testing deployment...${NC}"
if curl -f "http://${ALB_DNS}/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Service is responding correctly${NC}"
else
    echo -e "${YELLOW}⚠️  Service might still be starting up. Please wait a few minutes and try again.${NC}"
fi

echo -e "${GREEN}🎯 Deployment script completed!${NC}" 