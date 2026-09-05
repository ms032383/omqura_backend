#!/bin/bash
# AWS EC2 g4dn.xlarge Spot Instance Deployment Script for Neuro-AI Core Engine
# This script builds the docker image, pushes it to AWS ECR, and deploys it to a Spot instance.

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="123456789012" # Replace with actual AWS Account ID
ECR_REPO_NAME="neuro-ai-core"
IMAGE_TAG="latest"
INSTANCE_TYPE="g4dn.xlarge" # AWS T4 GPU 16 GB - ~$0.15/hr spot rate
SPOT_PRICE="0.20" # Max spot bid price

echo "=================================================="
echo "          AWS DEPLOYMENT PIPELINE START           "
echo "=================================================="

# 1. Authenticate Docker with AWS ECR
echo "Step 1: Authenticating Docker with AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# 2. Create Repository if it doesn't exist
echo "Step 2: Ensuring AWS ECR repository exists..."
aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION || true

# 3. Build & Tag Docker Image
echo "Step 3: Building and tagging Docker container..."
docker build -t $ECR_REPO_NAME:$IMAGE_TAG -f Dockerfile .
docker tag $ECR_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG

# 4. Push Image to ECR
echo "Step 4: Pushing Docker image to AWS ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG

# 5. Request Spot Instance with GPU configuration
# Below represents the AWS CLI call to request a Spot instance with a custom user data startup script.
echo "Step 5: Provisioning AWS Spot Instance ($INSTANCE_TYPE) with Docker installation..."

# Setup user data script to run at EC2 launch
cat << 'EOF' > user_data.sh
#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io awscli
sudo systemctl start docker
sudo systemctl enable docker

# Log in and pull the latest image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/neuro-ai-core:latest

# Run the container
docker run -d -p 80:8000 --name neuro_ai 123456789012.dkr.ecr.us-east-1.amazonaws.com/neuro-ai-core:latest
EOF

echo "Deployment configuration packaged!"
echo "AWS Spot CLI command generated:"
echo "aws ec2 request-spot-instances \\"
echo "  --spot-price-specification \"{SpotPrice=$SPOT_PRICE,InstanceType=$INSTANCE_TYPE}\" \\"
echo "  --launch-specification \"{ImageId=ami-0c7217cdde317cfec,KeyName=my-key-pair,SecurityGroups=[launch-wizard-1],UserData=$(base64 -w 0 user_data.sh)}\""

# Clean up local temporary file
rm -f user_data.sh

echo "=================================================="
echo "          AWS DEPLOYMENT SETUP COMPLETE           "
echo "=================================================="
