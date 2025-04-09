from sagemaker.jumpstart.model import JumpStartModel

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model = JumpStartModel(
    model_id="resnet-50-imagenet-classification",
    role=role
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)

result = predictor.predict({"inputs": "s3://your-bucket/sample.jpg"})
print(result)
