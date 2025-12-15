import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import yaml
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import gc
import random
from datetime import datetime


# === FUNCTION AND CLASS DEFINITIONS (OUTSIDE THE GUARD) ===

# Create a wrapper class that applies different transforms to subsets
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
        
# == BUILD MODEL ==
print("\n=== Building EfficientNet Model ===")

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(pretrained=True)
        
        # Get input features to classifier
        in_features = self.base_model.classifier[1].in_features
        
        # Replace classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)
    
    def freeze_base(self):
        """Freeze all base model layers except classifier"""
        for param in self.base_model.features.parameters():
            param.requires_grad = False
    
    def unfreeze_top_layers(self, num_layers=50):
        """Unfreeze top N layers for fine-tuning"""
        layers = list(self.base_model.features.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

# == CALCULATE CLASS WEIGHTS ==
def calculate_class_weights(dataset, num_classes, method = 'inverse', smoothing=0.0):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        method: 'inverse' or 'effective' (effective is better for severe imbalance)
        smoothing: 0.0 = no smoothing, 0.5 = moderate, 1.0 = uniform weights
    """
    class_counts = torch.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    if method == 'inverse':
        # Standard inverse frequency weighting
        total = class_counts.sum()
        class_weights = total / (num_classes * class_counts)
        
    elif method == 'effective':
        # For severe imbalance, use lower beta
        beta = 0.999  # Lower than 0.9999 for less aggressive weighting
        effective_num = 1.0 - torch.pow(beta, class_counts)
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / class_weights.sum() * num_classes
    
    # Apply smoothing
    if smoothing > 0:
        uniform_weights = torch.ones(num_classes)
        class_weights = (1 - smoothing) * class_weights + smoothing * uniform_weights
    
    print(f"\nClass weights (method={method}, smoothing={smoothing}):")
    for i, (weight, count) in enumerate(zip(class_weights, class_counts)):
        if count > 0:
            print(f"  Class {i:2d}: weight={weight:.4f}, samples={int(count)}")

    return class_weights

# == TRAINING FUNCTION ==
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# == VALIDATION FUNCTION ==
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


# === MAIN EXECUTION (INSIDE THE GUARD) ===

if __name__ == '__main__':

    SEED = 42  # Any number you like
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For multi-GPU
    
    # For MPS (local Mac)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)
    
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {SEED} for reproducibility")

    # Set device - MPS for M4 GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Change to your project directory
    os.chdir("/Users/jennylam/Documents/02 - Wissen/005 - IT/Rakuten_ECommerce_Proj/Rakuten-Ecommerce-Proj")
    print("Current working directory:", os.getcwd())

    # Load config
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)
    image_train_path_output = config["Paths"]["image_train_path_output"]

    # Settings
    USE_SUBSET = False
    SUBSET_RATIO = 0.10
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_WORKERS = 4

    print("\n=== Configuration ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Use subset: {USE_SUBSET}")
    if USE_SUBSET:
        print(f"Subset ratio: {SUBSET_RATIO}")
    
    # === DATA AUGMENTATION & PREPROCESSING ===
    # Training transforms (with augmentation)
    # == replaced by section below ==
    # train_transforms = transforms.Compose([
    #     transforms.Resize((IMG_SIZE, IMG_SIZE)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=20),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    # ])
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        
        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),  # Less common for product images
        transforms.RandomRotation(degrees=30),  # Increased from 20
        transforms.RandomResizedCrop(
            IMG_SIZE, 
            scale=(0.8, 1.0),  # Zoom in/out
            ratio=(0.9, 1.1)   # Slight aspect ratio changes
        ),
        
        # Color augmentations
        transforms.ColorJitter(
            brightness=0.3,    # Increased from 0.2
            contrast=0.3,      # Increased from 0.2
            saturation=0.3,    # NEW - color intensity changes
            hue=0.1           # NEW - slight color shifts
        ),
        
        # Additional augmentations
        transforms.RandomGrayscale(p=0.1),  # NEW - 10% chance grayscale
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),  # NEW - 30% chance blur (simulates focus issues)
        
        # Convert to tensor
        transforms.ToTensor(),
        
        # Normalize with ImageNet stats
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        
        # Post-tensor augmentation
        transforms.RandomErasing(
            p=0.25,              # 25% chance
            scale=(0.02, 0.2),   # Erase 2-20% of image
            ratio=(0.3, 3.3),    # Aspect ratio of erased region
            value='random'       # Fill with random noise
        )  # NEW - Cutout/Random Erasing (helps with occlusion)
    ])
    
    # Validation transforms (NO augmentation - just resize + normalize)
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("✓ Enhanced augmentation enabled")
    print("  - Geometric: horizontal/vertical flip, rotation, crop")
    print("  - Color: jitter (brightness, contrast, saturation, hue), grayscale")
    print("  - Quality: gaussian blur, random erasing")


    # === LOAD DATASETS ===
    print("\n=== Loading Datasets ===")

   # === LOAD DATASETS ===
    full_dataset = datasets.ImageFolder(root=image_train_path_output, transform=None)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    # === ANALYZE CLASS DISTRIBUTION ===
    print("\n=== Analyzing Class Distribution ===")
    all_labels = [label for _, label in full_dataset]
    all_labels_tensor = torch.tensor(all_labels)
    class_counts_full = torch.bincount(all_labels_tensor)
    
    for i, count in enumerate(class_counts_full):
        percentage = (count / len(full_dataset)) * 100
        print(f"Class {i:2d} ({class_names[i]:30s}): {count:5d} samples ({percentage:5.2f}%)")
    
    imbalance_ratio = class_counts_full.max().item() / class_counts_full.min().item()
    print(f"\n⚠️ Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # === SPLIT DATASET ===
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    if USE_SUBSET:
        subset_size = int(SUBSET_RATIO * len(full_dataset))
        full_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
        train_size = int(0.8 * subset_size)
        val_size = subset_size - train_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # === APPLY TRANSFORMS ===
    train_dataset = TransformDataset(train_subset, transform=train_transforms)
    val_dataset = TransformDataset(val_subset, transform=val_transforms)
    
    # === CREATE WEIGHTED SAMPLER (SOFTENED FOR SEVERE IMBALANCE) ===
    print("\n=== Creating Balanced Sampler ===")
    train_labels = [label for _, label in train_subset]
    train_labels_tensor = torch.tensor(train_labels)
    class_counts_train = torch.bincount(train_labels_tensor, minlength=num_classes)
    
    print("Training set class distribution:")
    for i, count in enumerate(class_counts_train):
        if count > 0:
            print(f"  Class {i}: {count} samples")
    
    imbalance = class_counts_train.max().item() / (class_counts_train[class_counts_train > 0].min().item())
    print(f"\nImbalance ratio: {imbalance:.2f}:1")
    
    # SOFTENED SAMPLING WEIGHTS (prevent extreme oversampling)
    # Instead of pure 1/count, use sqrt to soften
    class_weights_sampling = 1.0 / torch.sqrt(class_counts_train.float())  # ← SQRT HERE!
    class_weights_sampling[class_counts_train == 0] = 0
    
    # Apply additional smoothing
    smoothing = 0.3  # Reduce extremes
    uniform = torch.ones_like(class_weights_sampling)
    class_weights_sampling = (1 - smoothing) * class_weights_sampling + smoothing * uniform
    
    sample_weights = class_weights_sampling[train_labels_tensor]
    
    print(f"\nSampling weights range: {sample_weights.min():.4f} to {sample_weights.max():.4f}")
    print(f"Sampling weight ratio (max/min): {sample_weights.max()/sample_weights.min():.2f}:1")
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # === CREATE DATA LOADERS ===
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,  # Use sampler, not shuffle
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

  # === CALCULATE LOSS WEIGHTS (ALSO SOFTENED) ===
    print("\n=== Calculating Class Weights for Loss ===")
    
    # Use INVERSE method with SMOOTHING (not 'effective')
    class_weights_loss = calculate_class_weights(
        train_subset, 
        num_classes, 
        method='inverse',
        smoothing=0.5  # 50% smoothing for severe imbalance
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_loss)
    
    print(f"\nLoss weights range: {class_weights_loss.min():.4f} to {class_weights_loss.max():.4f}")
    print(f"Loss weight ratio (max/min): {class_weights_loss.max()/class_weights_loss.min():.2f}:1")

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')


    # === BUILD MODEL ===
    print("\n=== Building EfficientNet Model ===")
    model = EfficientNetClassifier(num_classes=num_classes, dropout=0.5)
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # === PHASE 1: Train with Frozen Base ===
    print("\n" + "="*60)
    print("PHASE 1: Training with Frozen Base Model")
    print("="*60)
    
    model.freeze_base()

    # Count trainable params after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (after freezing): {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-7
    )

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 12

    for epoch in range(25):
        print(f"\nEpoch {epoch+1}/25 - LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"⚠️ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model_phase1.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Clear memory
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()

    # === PHASE 2: Fine-tune Top Layers ===
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning Top Layers")
    print("="*60)
    
    model.unfreeze_top_layers(num_layers=100)

    # Count trainable params after unfreezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (after unfreezing): {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=3e-5)  # Lower learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-7
    )

    patience_counter = 0

    for epoch in range(40):
        print(f"\nEpoch {epoch+1}/40 - LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"⚠️ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model_final.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # === FINAL EVALUATION ===
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Create a folder name with current date and hour
    folder_name = datetime.now().strftime('%Y-%m-%d_%H_output')
    os.makedirs(folder_name, exist_ok=True)
    
    # Load best model
    checkpoint = torch.load('best_model_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    final_train_loss, final_train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    final_val_loss, final_val_acc, y_pred, y_true = validate_epoch(model, val_loader, criterion, device)
    
    final_report = os.path.join(folder_name, 'final_report.txt') 
    with open(final_report, 'w') as f:
        f.write(f"Random Seed: {SEED}%\n")
        f.write(f"Final Training Accuracy: {final_train_acc:.2f}%\n")
        f.write(f"Final Training Loss: {final_train_loss:.2f}%\n")
        f.write(f"Final Validation Accuracy: {final_val_acc:.2f}%\n")
        f.write(f"Final Validation Loss: {final_val_loss:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

    print(f"\n✓ Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"\n✓ Final Training Loss: {final_train_loss:.2f}%")
    print(f"\n✓ Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"\n✓ Final Validation Loss: {final_val_loss:.2f}%")
    print(f"✓ Best Validation Accuracy: {best_val_acc:.2f}%")

        # === FINAL EVALUATION ===
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)


    # Load best model
    checkpoint = torch.load('best_model_final.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    _, val_final_acc, y_pred_val, y_true_val = validate_epoch(model, val_loader, criterion, device)

    print(f"\n✓ Validation Accuracy: {val_final_acc:.2f}%")
    print(f"✓ Best Validation Accuracy (during training): {best_val_acc:.2f}%")

    # Validation classification report
    print("\n=== Validation Classification Report ===")
    report_val = classification_report(y_true_val, y_pred_val, target_names=class_names)
    print(report_val)

    class_report_val = os.path.join(folder_name, 'classification_report_VALIDATION.txt') 
    with open(class_report_val, 'w') as f:
        f.write(report_val)
    print("✓ Validation report saved")

    # Validation confusion matrix
    conf_matr_val = os.path.join(folder_name, 'confusion_matrix_VALIDATION.npy') 
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    np.save(conf_matr_val, cm_val)


    # === TEST SET EVALUATION ===
    print("\n" + "="*60)
    print("TEST SET EVALUATION (NEVER SEEN BEFORE)")
    print("="*60)
    
    # Load test dataset path from config
    image_test_path = config["Paths"]["image_test_path_output"]
    print(f"Loading test set from: {image_test_path}")
    
    # Load test dataset
    test_full_dataset = datasets.ImageFolder(root=image_test_path, transform=None)
    
    # Verify classes match
    assert test_full_dataset.classes == class_names, \
        f"Test classes {test_full_dataset.classes} don't match train classes {class_names}!"
    
    print(f"Test set: {len(test_full_dataset)} images")
    print(f"Test classes verified: {test_full_dataset.classes}")
    
    # Apply validation transforms (no augmentation)
    test_dataset = TransformDataset(test_full_dataset, transform=val_transforms)
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Keep 0 for macOS
        pin_memory=False
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    _, test_acc, y_pred_test, y_true_test = validate_epoch(model, test_loader, criterion, device)
    
    # Print results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Validation Accuracy: {val_final_acc:.2f}%")
    print(f"TEST Accuracy:       {test_acc:.2f}%")
    print(f"Difference:          {test_acc - val_final_acc:+.2f}%")
    print(f"{'='*60}")

    with open(final_report, 'w') as f:
        f.write(f"TEST Accuracy:       {test_acc:.2f}%\n")
    
    # Test classification report
    print("\n=== TEST Classification Report ===")
    report_test = classification_report(y_true_test, y_pred_test, target_names=class_names)
    print(report_test)
    
    class_report_test = os.path.join(folder_name, 'classification_report_TEST.txt') 
    with open(class_report_test, 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Validation Accuracy: {val_final_acc:.2f}%\n\n")
        f.write(report_test)
    print("✓ Test classification report saved to: classification_report_TEST.txt")
    
    # Test confusion matrix
    conf_matr_test = os.path.join(folder_name, 'confusion_matrix_TEST.npy') 
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    np.save(conf_matr_test, cm_test)
    print(f"✓ Test confusion matrix saved to: confusion_matrix_TEST.npy")
    
    # Per-class comparison
    print("\n=== Per-Class F1-Score Comparison ===")
    print(f"{'Class':<30} {'Val F1':>10} {'Test F1':>10} {'Diff':>10}")
    print("-" * 65)
    
    val_report_dict = classification_report(y_true_val, y_pred_val, target_names=class_names, output_dict=True)
    test_report_dict = classification_report(y_true_test, y_pred_test, target_names=class_names, output_dict=True)
    
    f1_comparison = os.path.join(folder_name, 'F1_Score_Comparison.txt') 
    with open( f1_comparison, 'w') as f:
        f.write(f"{'Class':<30} {'Val F1':>10} {'Test F1':>10} {'Diff':>10}\n")
        f.write(f"---------------------------------------------------------------------\n")
        for class_name in class_names:
            val_f1 = val_report_dict[class_name]['f1-score']
            test_f1 = test_report_dict[class_name]['f1-score']
            diff = test_f1 - val_f1
            print(f"{class_name:<30} {val_f1:>10.3f} {test_f1:>10.3f} {diff:>10.3f}")
            f.write(f"{class_name:<30} {val_f1:>10.3f} {test_f1:>10.3f} {diff:>10.3f}\n")
    print("✓ Test classification report saved to: F1_Score_Comparison.txt")

    # Update final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"✓ Best model: best_model_final.pth")
    print(f"✓ Validation accuracy: {val_final_acc:.2f}%")
    print(f"✓ TEST accuracy: {test_acc:.2f}%")
    print(f"✓ Validation report: classification_report_VALIDATION.txt")
    print(f"✓ TEST report: classification_report_TEST.txt")
    print(f"✓ Validation confusion matrix: confusion_matrix_VALIDATION.npy")
    print(f"✓ TEST confusion matrix: confusion_matrix_TEST.npy")



