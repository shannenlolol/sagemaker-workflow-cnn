import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

train_dir = '/opt/ml/input/data/train'
val_dir = '/opt/ml/input/data/validation'
model_dir = '/opt/ml/model'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
val_data = torchvision.datasets.ImageFolder(val_dir, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet50(pretrained=False, num_classes=2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
