#!/bin/bash

# Specify an algorithm name
repo_name=pytorch-extended-container
tag_name=pytorch1.8.0-training-cpu-py36-ubuntu18.04

echo Repository Name is ${repo_name}
echo Tag Name is ${tag_name}

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

echo Region is ${region}

# Make fullname combined by acoount, region, repo_name, tag_name
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${repo_name}:${tag_name}"

echo FullName is ${fullname}

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${repo_name}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
aws ecr create-repository --repository-name "${repo_name}" > /dev/null
fi

# Log into Docker
aws ecr get-login-password --region ${region} |docker login --username AWS --password-stdin ${fullname}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${repo_name} .
docker tag ${repo_name} ${fullname}

docker push ${fullname}