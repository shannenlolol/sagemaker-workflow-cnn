from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.p2.xlarge",
    framework_version="1.13",
    py_version="py39",
    output_path="s3://your-bucket-name/output"
)

estimator.fit({
    "train": TrainingInput("s3://your-bucket-name/train", content_type="application/x-image"),
    "validation": TrainingInput("s3://your-bucket-name/validation", content_type="application/x-image")
})
