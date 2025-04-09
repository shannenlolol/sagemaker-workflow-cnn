from sagemaker.pytorch import PyTorchModel

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model = PyTorchModel(
    model_data='s3://your-bucket/output/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.13',
    py_version='py39'
)

transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://your-bucket/predictions'
)

transformer.transform(
    data='s3://your-bucket/input',
    content_type='application/x-image',
    split_type='None'
)
