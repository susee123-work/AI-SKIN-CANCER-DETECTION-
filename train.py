# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from calibration import ModelWithTemperature
import numpy as np
from sklearn.metrics import roc_auc_score

# ==== Paths ====
train_dir = "dataset/train"
val_dir = "dataset/val"

# ==== Transforms ====
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==== Datasets & Weighted Sampler ====
train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

class_counts = np.bincount(train_ds.targets)
weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [weights[y] for y in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_dl = DataLoader(train_ds, batch_size=16, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

# ==== Device ====
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ==== Model ====
class SkinCancerNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_feats = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feats, num_classes)
        )

    def forward(self, x):
        return self.base(x)

model = SkinCancerNet().to(device)

# ==== Loss & Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== Training Loop ====
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_dl):.4f}")

# ==== Validation ====
model.eval()
y_true, y_probs = [], []
with torch.no_grad():
    for x, y in val_dl:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)[:, 1]
        y_true.extend(y.numpy())
        y_probs.extend(probs.cpu().numpy())

auc = roc_auc_score(y_true, y_probs)
best_thresh = 0.5
print(f"Validation ROC AUC: {auc:.4f}")

# ==== Temperature Scaling ====
model_temp = ModelWithTemperature(model)
model_temp.calibrate(val_dl, device=device)

# ==== Save Model & Config ====
torch.save(model_temp.model.state_dict(), "best_model.pth")
torch.save({"threshold": best_thresh}, "config.pth")
print("✅ Model and threshold saved successfully.")
