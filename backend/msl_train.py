import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
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

# Training transforms: moderate augmentation only
# NOTE: NO RandomHorizontalFlip — flipping changes the semantic meaning of signs
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Eval transforms: no augmentation
eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = '../msl_dataset/AlphabetsV2/AlphabetsV2'
full_dataset = torchvision.datasets.ImageFolder(root=dataset_path)
num_classes  = len(full_dataset.classes)

# 80/20 split — maximise training data (dataset is small: ~114 images/class for train)
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

# --- 2. THE CNN MODEL ---
# 3 conv blocks (16->32->64 filters) + BatchNorm + Dropout(0.3)
# Input 128x128 -> pool -> 64x64 -> pool -> 32x32 -> pool -> 16x16
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),    # Reduced from 0.5 — prevents underfitting on small dataset
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

# --- 3. SETUP ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware: {device}\n")

model     = SignLanguageCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. TRAINING LOOP ---
epochs = 20
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

    print(f"Epoch [{epoch+1:2d}/{epochs}]  Loss: {avg_loss:.4f}  Val Accuracy: {100*val_correct/val_total:.2f}%")

# --- 5. SAVE ---
print("\nTraining complete!")
torch.save(model.state_dict(), 'msl_alphabet_model.pth')
print("Model saved to msl_alphabet_model.pth")

# --- 6. FINAL TEST ACCURACY ---
print("\nEvaluating on test set...")
model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")