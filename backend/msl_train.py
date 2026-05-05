import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset

# --- THE DATASET WRAPPER ---
# A structured way to ensure ONLY training data is augmented, keeping test data pure.
class MSLDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index] # Get raw PIL image and label
        if self.transform:
            x = self.transform(x) # Apply the specific transform
        return x, y

    def __len__(self):
        return len(self.subset)

# --- 1. THE DATA ---
print("Preparing MSL dataset with Data Augmentation...")

# Training Transforms: Add random rotations and lighting shifts
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

# Testing Transforms: Pure tensors, no random changes
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

dataset_path = '../msl_dataset/AlphabetsV2/AlphabetsV2' 

# Load raw dataset
full_dataset = torchvision.datasets.ImageFolder(root=dataset_path)
num_classes = len(full_dataset.classes)

# Split: 80% Train, 20% Test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

# Apply the strict transforms
train_dataset = MSLDataset(train_subset, transform=train_transform)
test_dataset = MSLDataset(test_subset, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(train_dataset)} training images and {len(test_dataset)} testing images.")
print(f"Classes detected: {num_classes} classes -> {full_dataset.classes}\n")

# --- 2. THE CNN MODEL ---
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# --- 3. THE SETUP ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Hardware accelerated on: {device}\n")

model = SignLanguageCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. THE TRAINING LOOP ---
epochs = 15 # Increased to give the model time to learn the variations
print("Starting training on Apple Silicon (MPS)...")

for epoch in range(epochs):
    running_loss = 0.0
    model.train() 
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}')

print("\nTraining complete!")

# --- 5. SAVE THE MODEL ---
save_path = 'msl_alphabet_model.pth'
torch.save(model.state_dict(), save_path)
print(f"Model saved successfully to {save_path}")