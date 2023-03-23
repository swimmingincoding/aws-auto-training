import argparse
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

def main(args):
    # Set up the Environment
    sagemaker_session = sagemaker.Session()

    # IAM role setting
    role = get_execution_role()

    # Create a training job using the PyTorch Estimator
    estimator = PyTorch(image_uri=args.ecr_image,
                        entry_point=args.entry_point,
                        source_dir=args.source_dir,
                        role=role,
                        train_instance_count=1,
                        train_instance_type=args.train_instance_type, # ml.c5.xlarge, ml.p2.xlarge
                        hyperparameters={
                            "epochs": args.epochs,
                            "backend": "gloo"
                        })

    # # Create a training job using the PyTorch Estimator
    # estimator = PyTorch(entry_point="train.py",
    #                     role=role,
    #                     framework_version="1.8.0",
    #                     py_version="py36",
    #                     train_instance_count=1,
    #                     train_instance_type=args.train_instance_type, # ml.c5.xlarge, ml.p2.xlarge
    #                     hyperparameters={
    #                         "epochs": args.epochs,
    #                         "backend": "gloo"
    #                     })

    # Calling `fit`
    estimator.fit({"train": f"{args.data_dir}"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model parameters
    parser.add_argument("--epochs", type=int, default=5, metavar="N",
                        help="number of epochs to train (default: 5)")
    
    # Container environment
    parser.add_argument("--ecr-image", type=str, default="503319961193.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-extended-container:pytorch1.8.0-training-cpu-py36-ubuntu18.04",
                        help="PyTorch Extended Container from AWS PyTorch Container")
    parser.add_argument("--entry-point", type=str, default="/home/ec2-user/SageMaker/aws-auto-training/train.py")
    parser.add_argument("--source-dir", type=str, default="/home/ec2-user/SageMaker/aws-auto-training/")
    parser.add_argument("--train-instance-type", type=str, default="ml.c5.xlarge")
    parser.add_argument("--data-dir", type=str, default="s3://sagemaker-pytorch-metal/data/")

    main(parser.parse_args())