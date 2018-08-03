#!/bin/bash

job_name=$1
if [ -z $job_name ] 
then
    echo 'Provide model name'
    exit 0
fi 
echo 'Creating training job '$1

training_image="<replace_me>.dkr.ecr.eu-west-1.amazonaws.com/robocars:1.4.1-gpu-py3"
iam_role_arn="arn:aws:iam::<replace_me>:role/service-role/<replace_me>"

aws sagemaker create-training-job \
    --training-job-name $job_name \
    --hyper-parameters '{ "sagemaker_region": "\"eu-west-1\"", "with_slide": "true" }' \
    --algorithm-specification TrainingImage=$training_image,TrainingInputMode=File \
    --role-arn $iam_role_arn \
    --input-data-config '[{ "ChannelName": "train", "DataSource": { "S3DataSource": { "S3DataType": "S3Prefix", "S3Uri": "s3://<replace_me>", "S3DataDistributionType": "FullyReplicated" }} }]' \
    --output-data-config S3OutputPath=s3://<replace_me> \
    --resource-config InstanceType=ml.p2.xlarge,InstanceCount=1,VolumeSizeInGB=1 \
    --stopping-condition MaxRuntimeInSeconds=1800
