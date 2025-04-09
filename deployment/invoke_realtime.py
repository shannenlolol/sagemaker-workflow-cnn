from sagemaker.predictor import Predictor
from PIL import Image
import io
import numpy as np

# Load your image
image = Image.open("cat.jpg").convert("RGB")
image = image.resize((224, 224))
image_bytes = io.BytesIO()
image.save(image_bytes, format="JPEG")
image_bytes = image_bytes.getvalue()

# Connect to your deployed endpoint
predictor = Predictor(endpoint_name="your-endpoint-name")

# Inference
result = predictor.predict(image_bytes, content_type="application/x-image")
print("Inference Result:", result)
