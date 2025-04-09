from sagemaker.pytorch import PyTorchModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model = PyTorchModel(
    model_data='s3://your-bucket/output/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.13',
    py_version='py39'
)

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5
)

predictor = model.deploy(
    serverless_inference_config=serverless_config
)

response = predictor.predict({"inputs": "example input"})
print(response)
