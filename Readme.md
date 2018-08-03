# Purpose

Run DIY Robocars model training as Sagemaker (https://aws.amazon.com/fr/sagemaker/) task. Estimated cost for one training (as of August 2018): 0.50 EUR

# Build images

- Build base image:

``` 
docker build -t robocars-base:1.4.1-gpu-py3 -f Dockerfile_base.gpu .
```

- Build model image:

``` 
docker build -t robocars:1.4.1-gpu-py3 -f Dockerfile.gpu .
```

# Prepare training (once)

- Create a S3 bucket for your tubes. You can use the same for model output or create another bucker for output
- Create an AWS docker registry and push your model image to it. Docker hub registry is not supported

``` 
docker tag robocars:1.4.1-gpu-py <replace_me>.dkr.ecr.eu-west-1.amazonaws.com/robocars:1.4.1-gpu-py3
# you should have AWS SDK installed and login to docker
docker push <replace_me>.dkr.ecr.eu-west-1.amazonaws.com/robocars:1.4.1-gpu-py3
``` 

# Run training

- Copy your tubes to your S3 bucket. All tubes in the bucket will be used for training so make sure you keep only relevant files. We recommend to zip your tubes before upload. The training package will unzip them.
- Create a training job on AWS Sagemaker. Use create_job.sh script after replacing relevant parameters

```
#!/bin/bash

#usage: create_job.sh some_job_unique_name
job_name=$1
if [ -z $job_name ] 
then
    echo 'Provide job unique name'
    exit 0
fi 
echo 'Creating training job '$1

aws sagemaker create-training-job \
    --training-job-name $job_name \
    --hyper-parameters '{ "sagemaker_region": "\"eu-west-1\"", "with_slide": "true" }' \
    --algorithm-specification TrainingImage="<replace_me>.dkr.ecr.eu-west-1.amazonaws.com/robocars:1.4.1-gpu-py3",TrainingInputMode=File \
    --role-arn "<your_iam_sagemaker_role>" \
    --input-data-config '[{ "ChannelName": "train", "DataSource": { "S3DataSource": { "S3DataType": "S3Prefix", "S3Uri": "s3://<your_input_bucket>", "S3DataDistributionType": "FullyReplicated" }} }]' \
    --output-data-config S3OutputPath=s3://<your_output_bucket> \
    --resource-config InstanceType=ml.p2.xlarge,InstanceCount=1,VolumeSizeInGB=1 \
    --stopping-condition MaxRuntimeInSeconds=1800
```

- Keep an eye on job progression on AWS Sagemaker. Once finished your model is copied into the destination bucket.

# About AWS Sagemaker

Sagemaker provide on-demand model computing and serving. Standard algorithms can be used and on-demande Jupyter notebooks are available. However, as any hosted service, tensorflow versions are updated frequently which is not manageable because compatible versions might not be available on RaspberryPi. Sagemaker also allow "Bring Your Own Algorithm" by using a docker image for training. The resulting container must comply to Sagemaker constraints.

Input and output data are mapped to S3 buckets: at container start, input data is copied to ``` /opt/ml/input/data/train ``` and at the end of training data in ```/opt/ml/``` is copied back to S3. 

Hyperparameters can be sent at job creation time and accessed by training code (example: ```env.hyperparameters.get('with_slide', False)```)

# Which Tensorflow version should I pick ?

Version 1.4.1 model is compatible with 1.8.0 tensorflow runtime

Version 1.8.0 model is not compatible with previous tensorflow runtimes

