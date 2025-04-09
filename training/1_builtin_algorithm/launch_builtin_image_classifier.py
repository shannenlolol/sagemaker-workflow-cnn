from sagemaker import image_uris, Estimator
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"
image_uri = image_uris.retrieve("image-classification", "us-west-2")

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.p2.xlarge",
    output_path="s3://your-bucket-name/output",
    hyperparameters={
        "num_layers": 18,
        "image_shape": "3,224,224",
        "num_classes": 2,
        "num_training_samples": 1000,
        "mini_batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.01,
        "use_pretrained_model": 1,
    }
)

estimator.fit({
    "train": TrainingInput("s3://your-bucket-name/train", content_type="application/x-image"),
    "validation": TrainingInput("s3://your-bucket-name/validation", content_type="application/x-image")
})
