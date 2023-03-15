import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

# Set up the Environment
sagemaker_session = sagemaker.Session()

# IAM role setting
role = get_execution_role()

# Create a training job using the PyTorch Estimator
estimator = PyTorch(entry_point="train.py",
                    role=role,
                    framework_version="1.8.0",
                    py_version="py3",
                    train_instance_count=1,
                    train_instance_type="ml.c5.xlarge",
                    hyperparameters={
                        "epochs": 5,
                        "backend": "gloo"
                    })

# Calling `fit`
data_path = "s3://sagemaker-pytorch-metal/data/"
estimator.fit({"train": f"{data_path}"})