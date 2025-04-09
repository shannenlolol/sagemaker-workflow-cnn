from sagemaker.pytorch import PyTorchModel

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model = PyTorchModel(
    model_data='s3://your-bucket/output/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.13',
    py_version='py39'
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

result = predictor.predict({"inputs": "example input"})
print(result)
