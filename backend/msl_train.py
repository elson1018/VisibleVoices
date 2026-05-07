import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split, Dataset

# --- THE DATASET WRAPPER ---
class MSLDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# --- 1. THE DATA ---
print("Preparing MSL dataset...")

# MobileNetV2 was trained on ImageNet with 224x224 — match that for best transfer
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    # ImageNet normalisation values (required for pretrained MobileNetV2)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_path = '../msl_dataset/AlphabetsV2/AlphabetsV2'
full_dataset = torchvision.datasets.ImageFolder(root=dataset_path)
num_classes  = len(full_dataset.classes)

# 80/10/10 split
train_size = int(0.80 * len(full_dataset))
val_size   = int(0.10 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size
train_subset, val_subset, test_subset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset = MSLDataset(train_subset, transform=train_transform)
val_dataset   = MSLDataset(val_subset,   transform=eval_transform)
test_dataset  = MSLDataset(test_subset,  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

print(f"Split — Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
print(f"Classes: {num_classes} -> {full_dataset.classes}\n")

# --- 2. TRANSFER LEARNING: MobileNetV2 ---
# MobileNetV2 was pretrained on 1.2M ImageNet images.
# It already knows how to detect edges, textures, shapes — far better than
# a scratch CNN trained on only ~114 images/class.
print("Loading pretrained MobileNetV2...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze all pretrained layers — we don't want to destroy what it already learned
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier with one for our 26 MSL classes
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

# Unfreeze the last 3 feature blocks for domain adaptation
# (lets the model adjust its high-level features for hand signs)
for param in model.features[-4:].parameters():
    param.requires_grad = True

print("Trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} / "
      f"{sum(p.numel() for p in model.parameters()):,} total\n")

# --- 3. SETUP ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware: {device}\n")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Use different learning rates: small LR for fine-tuned layers, larger for new classifier
optimizer = optim.Adam([
    {'params': model.features[-4:].parameters(), 'lr': 1e-4},  # Fine-tune pretrained layers slowly
    {'params': model.classifier.parameters(),     'lr': 1e-3},  # New classifier can learn faster
])

# --- 4. TRAINING LOOP ---
epochs = 20
best_val_acc = 0.0
print("Starting training...\n")

for epoch in range(epochs):
    # Train
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validate
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    marker  = ' ← best' if val_acc > best_val_acc else ''
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'msl_alphabet_model.pth')  # Save best checkpoint

    print(f"Epoch [{epoch+1:2d}/{epochs}]  Loss: {avg_loss:.4f}  Val Accuracy: {val_acc:.2f}%{marker}")

print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")
print("Best model already saved to msl_alphabet_model.pth")

# --- 5. FINAL TEST ACCURACY ---
print("\nEvaluating best model on test set...")
model.load_state_dict(torch.load('msl_alphabet_model.pth', map_location=device))
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")