# SageMaker End-to-End Workflow (ResNet-50 CNN)

This repository demonstrates **end-to-end training, deployment, evaluation, monitoring, and automation** of a **ResNet-50 CNN** using Amazon SageMaker.

---

## Project Structure Overview

| Directory      | Description                                       |
| -------------- | ------------------------------------------------- |
| `training/`    | All training methods using SageMaker              |
| `deployment/`  | Multiple deployment strategies post-training      |
| `evaluation/`  | Script to compute metrics like accuracy, F1-score |
| `monitoring/`  | Script to monitor endpoint health                 |
| `pipelines/`   | SageMaker Pipeline script for MLOps               |
| `inference.py` | Model serving logic for real-time/batch inference |

---

## Dataset Format

Amazon SageMaker expects the dataset in this directory layout (uploaded to S3):

```
s3://your-bucket/
├── train/
│   ├── class_a/
│   └── class_b/
└── validation/
    ├── class_a/
    └── class_b/
```

Each folder (`class_a`, `class_b`, etc.) should contain `.jpg` or `.png` files of that class.

---

## Setup

1. Make sure your environment has:

   - AWS CLI configured (`aws configure`)
   - SageMaker IAM execution role
   - Python 3.8+
   - Docker (for BYOC option)

2. Clone the repo:

```bash
git clone https://github.com/shannenlolol/sagemaker-workflow-cnn.git
cd sagemaker-workflow-cnn
```

3. Replace:
   - `role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/..."` with your actual SageMaker role
   - `s3://your-bucket-name/` with your dataset bucket

---

## Step 1: Train the Model

| Folder                            | Method                          | Description                                           |
| --------------------------------- | ------------------------------- | ----------------------------------------------------- |
| `training/1_builtin_algorithm/`   | Built-in Algorithm              | Zero-code training using SageMaker’s image classifier |
| `training/2_script_mode/`         | Script Mode                     | PyTorch container + your training script              |
| `training/3_custom_script_mode/`  | Script Mode with Dependencies   | Add a `requirements.txt` to install extra libraries   |
| `training/4_byoc/`                | Bring Your Own Container (BYOC) | Full custom Docker image and training environment     |
| `training/5_pretrained_finetune/` | Pretrained Fine-Tuning          | Fine-tune pretrained ResNet-50 on your dataset        |

Output:
A trained model file, usually saved in S3 as:

```bash
s3://your-bucket/output/model.tar.gz
```
---
## Step 2: Inference Script for Deployment

The `inference.py` file defines how SageMaker loads the model and serves predictions.

| Function     | Purpose                                     |
| ------------ | ------------------------------------------- |
| `model_fn`   | Loads `model.pth` and initialises ResNet-50 |
| `input_fn`   | Converts incoming images to tensors         |
| `predict_fn` | Runs inference                              |
| `output_fn`  | Formats output as JSON                      |

---


## Step 3: Deploy the Model to an Endpoint

| Script                                 | Method                 | Description                                            |
| -------------------------------------- | ---------------------- | ------------------------------------------------------ |
| `deployment/1_realtime_endpoint.py`    | Real-Time Inference    | Low-latency HTTPS endpoint using `PyTorchModel`        |
| `deployment/2_batch_transform.py`      | Batch Transform        | Inference over large S3 datasets (offline)             |
| `deployment/3_serverless_inference.py` | Serverless Inference   | Pay-per-request prediction, no EC2 management          |
| `deployment/4_async_inference.py`      | Asynchronous Inference | Queue-based prediction for large or long jobs          |
| `deployment/5_jumpstart_model.py`      | JumpStart Pretrained   | Deploy ResNet-50 pretrained model from AWS JumpStart   |
| `deployment/6_jumpstart_estimator.py`  | JumpStart Fine-Tuning  | Fine-tune JumpStart's ResNet-50 model on your own data |

---

## Step 4: Make Predictions (Client Script)

Client-side file:
```bash
deployment/invoke_realtime.py
```

This file:
- Loads test input (e.g. an image)

- Sends it to the SageMaker endpoint

- Receives the prediction (inference result)

---

## Step 5: Evaluation

Evaluate your model using:

```bash
python evaluation/evaluate.py --model-path model.pth --data-dir /path/to/validation
```

Outputs include:

- Accuracy
- Precision, Recall, F1-score
- Full classification report (via `sklearn.metrics`)

---

## Additional Step 1: Pipelines for MLOps

The `pipelines/build_pipeline.py` automates:

- Model training
- Parameter tracking
- Pipeline versioning

Use SageMaker Studio or SDK to trigger and monitor this pipeline.

---

## Additional Step 2: Monitoring Your Endpoints 

Check endpoint health using:

```bash
python monitoring/monitor_endpoint.py
```

Returns:

- Endpoint status (`InService`, `Failed`, etc.)
- Useful for CI/CD or production monitoring

---

## License

MIT License

---

## Acknowledgements

- [Amazon SageMaker](https://aws.amazon.com/sagemaker/)
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/)
