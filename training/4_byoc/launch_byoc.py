from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

estimator = Estimator(
    image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/my-custom-cnn",
    role=role,
    instance_count=1,
    instance_type="ml.p2.xlarge",
    output_path="s3://your-bucket-name/output"
)

estimator.fit({
    "train": TrainingInput("s3://your-bucket-name/train", content_type="application/x-image"),
    "validation": TrainingInput("s3://your-bucket-name/validation", content_type="application/x-image")
})
