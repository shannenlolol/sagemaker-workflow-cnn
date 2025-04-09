from sagemaker.pytorch import PyTorchModel
from sagemaker.async_inference import AsyncInferenceConfig

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model = PyTorchModel(
    model_data='s3://your-bucket/output/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.13',
    py_version='py39'
)

async_config = AsyncInferenceConfig(
    output_path="s3://your-bucket/async-output/"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    async_inference_config=async_config
)

response = predictor.predict({"inputs": "example input"}, async_inference=True)
print(response)
