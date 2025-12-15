# ==================================================
# BASIC CNN FROM SCRATCH - LOCAL COMPUTER
# ==================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import yaml
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import gc
import random

# === SET RANDOM SEEDS ===
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)
# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"✓ Random seed set to {SEED}")

# === DEVICE SETUP ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# === CONFIGURATION ===
os.chdir("/Users/jennylam/Documents/02 - Wissen/005 - IT/Rakuten_ECommerce_Proj/Rakuten-Ecommerce-Proj")
print("Current working directory:", os.getcwd())

with open("./config.yaml", "r") as f:
    config = yaml.safe_load(f)
image_train_path_output = config["Paths"]["image_train_path_output"]
image_test_path = config["Paths"]["image_test_path_output"]

BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 0  # macOS compatibility

print("\n=== Configuration ===")
print(f"Batch size: {BATCH_SIZE}")
print(f"Image size: {IMG_SIZE}")
print(f"Seed: {SEED}")

# === DATA AUGMENTATION ===
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("✓ Transforms defined")

# === WRAPPER CLASS FOR TRANSFORMS ===
class TransformDataset(torch.utils.data.Dataset):
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

# === LOAD DATASETS ===
print("\n=== Loading Datasets ===")

# Load train dataset
train_full_dataset = datasets.ImageFolder(root=image_train_path_output, transform=None)
num_classes = len(train_full_dataset.classes)
class_names = train_full_dataset.classes

print(f"Training images: {len(train_full_dataset)}")
print(f"Number of classes: {num_classes}")

# Load test dataset
test_full_dataset = datasets.ImageFolder(root=image_test_path, transform=None)
print(f"Test images: {len(test_full_dataset)}")

# Verify classes match
assert test_full_dataset.classes == class_names, "Test classes don't match training!"
print("✓ Classes verified")

# === SPLIT TRAIN INTO TRAIN/VAL (SAME AS EFFICIENTNET) ===
train_size = int(0.8 * len(train_full_dataset))
val_size = len(train_full_dataset) - train_size

train_subset, val_subset = random_split(
    train_full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)  # Ensure same split
)

print(f"\nDataset splits:")
print(f"  Train: {len(train_subset)}")
print(f"  Validation: {len(val_subset)}")
print(f"  Test: {len(test_full_dataset)}")

# Apply transforms
train_dataset = TransformDataset(train_subset, transform=train_transforms)
val_dataset = TransformDataset(val_subset, transform=val_transforms)
test_dataset = TransformDataset(test_full_dataset, transform=val_transforms)

# === CREATE DATALOADERS ===
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

print(f"Batches - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

# === DEFINE BASIC CNN MODEL ===
class BasicCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(BasicCNN, self).__init__()
        
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x

# === BUILD MODEL ===
print("\n=== Building Basic CNN Model ===")
model = BasicCNN(num_classes=num_classes, dropout=0.5)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# === LOSS AND OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.2, patience=7, min_lr=1e-7
)

# === TRAINING FUNCTIONS ===
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# === TRAINING LOOP ===
print("\n" + "="*60)
print("TRAINING BASIC CNN FROM SCRATCH")
print("="*60)

best_val_acc = 0.0
patience_counter = 0
PATIENCE = 15
EPOCHS = 50

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS} - LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
    
    print(f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_basic_cnn.pth')
        print(f"✓ Saved! Best: {val_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Clear memory
gc.collect()
if device.type == 'mps':
    torch.mps.empty_cache()

# === FINAL EVALUATION ===
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load('best_basic_cnn.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Train
_, train_final_acc = train_epoch(model, train_loader, criterion, optimizer, device)

# Validation
_, val_final_acc, y_pred_val, y_true_val = validate_epoch(model, val_loader, criterion, device)

# Test
_, test_acc, y_pred_test, y_true_test = validate_epoch(model, test_loader, criterion, device)

print(f"\n{'='*60}")
print("BASIC CNN RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {train_final_acc:.2f}%")
print(f"Validation Accuracy: {val_final_acc:.2f}%")
print(f"TEST Accuracy:       {test_acc:.2f}%")
print(f"{'='*60}")

# Classification reports
report_test = classification_report(y_true_test, y_pred_test, target_names=class_names)
print("\n=== TEST Classification Report ===")
print(report_test)

with open('classification_report_BasicCNN_TEST.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")
    f.write(f"Validation Accuracy: {val_final_acc:.2f}%\n\n")
    f.write(report_test)

# Confusion matrix
cm_test = confusion_matrix(y_true_test, y_pred_test)
np.save('confusion_matrix_BasicCNN_TEST.npy', cm_test)

print("\n✓ Basic CNN training complete!")
print(f"✓ Model saved: best_basic_cnn.pth")
print(f"✓ Test accuracy: {test_acc:.2f}%")

