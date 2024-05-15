### Dagshub repo link
https://dagshub.com/sraj28383/mlflow_basics

### For Environment variables setup
conda env config vars set MLFLOW_TRACKING_URI=https://dagshub.com/sraj28383/mlflow_basics.mlflow MLFLOW_TRACKING_USERNAME=sraj28383 MLFLOW_TRACKING_PASSWORD=1e30d8370940cc032c8380ef5f9dfc994190ee6f

---

### MLFLOW ON AWS

### AWS Setup:
1. Login to AWS console
2. Create IAM user with AdministratorAccess
3. Export credentials in AWS CLI by running "aws configure"
4. Create S3 bucket
5. Create EC2 machine (ubuntu) & add security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell

#set aws credentials
aws configure

mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflow-test-23
```

```bash
conda env config vars set MLFLOW_TRACKING_URI="your URI"
```