from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

estimator = JumpStartEstimator(
    model_id="pytorch-resnet-50-imagenet-classification",
    role=role,
    instance_type="ml.p3.2xlarge",
    instance_count=1
)

estimator.fit({
    "training": TrainingInput("s3://your-bucket/train/"),
    "validation": TrainingInput("s3://your-bucket/validation/")
})
