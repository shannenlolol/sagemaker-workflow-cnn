from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline_context import PipelineSession

pipeline_session = PipelineSession()
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

model_path_param = ParameterString(name="ModelOutputPath", default_value="s3://your-bucket/output")

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    framework_version="1.13",
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    output_path=model_path_param,
    py_version="py39",
)

step_train = TrainingStep(
    name="TrainCNN",
    estimator=estimator,
    inputs={"train": "s3://your-bucket/train", "validation": "s3://your-bucket/validation"},
)

pipeline = Pipeline(
    name="CNNTrainingPipeline",
    parameters=[model_path_param],
    steps=[step_train],
    sagemaker_session=pipeline_session,
)
pipeline.upsert(role_arn=role)
