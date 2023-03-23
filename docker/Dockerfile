# SageMaker PyTorch Image
FROM 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/pytorch-training:1.8.0-cpu-py36-ubuntu18.04

ENV PATH="/opt/ml/code:${PATH}"

# Thie environment variable is used by the SageMaker, use the /code subdirectory to store your user code.
COPY /home/ec2-user/SageMaker/train.py /opt/ml/code/train.py
COPY /home/ec2-user/SageMaker/requirements.txt /opt/ml/code/requirements.txt

# Intall python packages
RUN pip install -r /opt/ml/code/requirements.txt

# Defines train.py as script entrypoints
ENV SAGEMAKER_PROGRAM train.py