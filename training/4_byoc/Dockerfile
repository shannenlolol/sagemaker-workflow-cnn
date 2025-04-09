FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime
RUN pip install torchvision
COPY train.py .
ENTRYPOINT ["python", "train.py"]
