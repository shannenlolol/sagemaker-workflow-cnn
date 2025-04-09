import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--data-dir', type=str, required=True)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder(args.data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
model.load_state_dict(torch.load(args.model_path))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, target_names=dataset.classes))
