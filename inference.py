import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Load model
def model_fn(model_dir):
    import torchvision.models as models
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model

# Preprocess input
def input_fn(request_body, request_content_type):
    if request_content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)
    raise ValueError("Unsupported content type")

# Run inference
def predict_fn(input_data, model):
    with torch.no_grad():
        output = model(input_data)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities.tolist()

# Format output
def output_fn(prediction, content_type):
    if content_type == "application/json":
        return {"probabilities": prediction}
    raise ValueError("Unsupported response content type")
