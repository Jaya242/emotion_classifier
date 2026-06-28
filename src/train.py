"""
FER2013 emotion classifier — Day 1 setup + dataset exploration.
"""

from datasets import load_dataset
from PIL import Image                                                                                                                                       
import io
import matplotlib.pyplot as plt

  # 1. Load
print("Loading FER2013 from HuggingFace...")
dataset = load_dataset("Jeneral/fer-2013")

  # 2. Structure
print("\n=== Dataset structure ===")
print(dataset)

  # Inspect feature types — important debugging line
print("\n=== Feature types ===")
print(dataset["train"].features)

  # 3. Use the actual column names from THIS dataset
train = dataset["train"]
LABEL_COL = "labels"
IMG_COL = "img_bytes"

  # 4. Get class names. If the dataset has a ClassLabel feature, it has .names.
  #    Otherwise fall back to standard FER2013 ordering.
label_feature = train.features[LABEL_COL]
if hasattr(label_feature, "names"):
    class_names = label_feature.names
    print(f"\n=== Classes ({len(class_names)}) — from dataset ===")
else:
    class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    print(f"\n=== Classes (FER2013 standard order) ===")
print(class_names)

  # 5. Splits
print("\n=== Splits ===")                                                                                                                                   
for split_name in dataset.keys():
    print(f"  {split_name}: {len(dataset[split_name])} samples")

  # 6. Inspect one sample (decode bytes → PIL Image)
print("\n=== Sample 0 ===")
sample = train[0]
print(f"  Keys: {list(sample.keys())}")
print(f"  img_bytes type: {type(sample[IMG_COL])}")
print(f"  Label: {sample[LABEL_COL]} → {class_names[sample[LABEL_COL]]}")

def decode_img(sample):
    """Helper: decode img_bytes → PIL Image (handles both bytes and already-decoded cases)."""
    img_data = sample[IMG_COL]
    if isinstance(img_data, bytes):
        return Image.open(io.BytesIO(img_data))
    return img_data  # already a PIL Image

img = decode_img(sample)
print(f"  Decoded image size: {img.size}")
print(f"  Mode: {img.mode}")  # likely 'L' for grayscale or 'RGB'

  # 7. Visualize 5 samples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
import random
random.seed(42)
indices = random.sample(range(len(train)), 5)
for ax_idx, sample_idx in enumerate(indices):
      sample = train[sample_idx]
      img = decode_img(sample)
      axes[ax_idx].imshow(img, cmap="gray")
      axes[ax_idx].set_title(class_names[sample[LABEL_COL]])
      axes[ax_idx].axis("off")
plt.tight_layout()
plt.savefig("sample_visualization.png", dpi=100)
plt.close()
print("\n✅ Saved sample_visualization.png — open it to see 5 example faces.")


  # ====================================================================
  # Block 3 — ResNet18 transfer learning setup
  # ====================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

  # Pick device — CPU on Mac (MPS exists but ResNet18 fine on CPU for now)
device = (
      torch.device("cuda") if torch.cuda.is_available()
      else torch.device("mps") if torch.backends.mps.is_available()
      else torch.device("cpu")
  )
print(f"\n=== Device: {device} ===")
  # ResNet18 expects 224x224 RGB. FER2013 is 48x48 grayscale.
  # We need transforms to convert.
# train_transform = transforms.Compose([
#       transforms.Grayscale(num_output_channels=3),      # 1 channel → 3 (duplicate)
#       transforms.Resize((224, 224)),                     # 48x48 → 224x224
#       transforms.RandomHorizontalFlip(),                 # data augmentation
#       transforms.ToTensor(),                             # PIL → tensor, scales to [0, 1]
#       transforms.Normalize(                              # ImageNet stats
#           mean=[0.485, 0.456, 0.406],
#           std=[0.229, 0.224, 0.225],
#       ),
# ])
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
])  
  # Test set: no augmentation, otherwise identical                                                                                                            
test_transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=3),
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

  # Custom collate to decode bytes + apply transform inside DataLoader
def collate_train(batch):
      images = torch.stack([train_transform(decode_img(s)) for s in batch])
      labels = torch.tensor([s[LABEL_COL] for s in batch])
      return images, labels

def collate_test(batch):                                                                                                                                    
      images = torch.stack([test_transform(decode_img(s)) for s in batch])
      labels = torch.tensor([s[LABEL_COL] for s in batch])
      return images, labels

  # DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(
      train, batch_size=BATCH_SIZE, shuffle=True,
      collate_fn=collate_train, num_workers=0,
  )
test_loader = DataLoader(
      dataset["test"], batch_size=BATCH_SIZE, shuffle=False,
      collate_fn=collate_test, num_workers=0,
  )
print(f"=== DataLoaders ready ===")
print(f"  train batches: {len(train_loader)}")
print(f"  test batches:  {len(test_loader)}")

  # Build model
print(f"\n=== Loading ResNet18 ===")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
  # Swap final layer: 1000 ImageNet classes → 7 emotions
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model.to(device)
print(f"  Final layer: {model.fc}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

  # Smoke test: one forward pass on one batch to verify shapes
print(f"\n=== Smoke test: one forward pass ===")
images, labels = next(iter(train_loader))
print(f"  Batch images: {images.shape}")        # Expect [32, 3, 224, 224]
print(f"  Batch labels: {labels.shape}")        # Expect [32]
with torch.no_grad():
    outputs = model(images.to(device))
print(f"  Model output: {outputs.shape}")       # Expect [32, 7]
print("✅ Shapes correct — pipeline is wired end-to-end.")



 # ====================================================================
  # Block 4 — Train ONE epoch (Day 1 goal: pipeline works end-to-end)
  # ====================================================================
import torch.optim as optim
from tqdm import tqdm
import os

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 8
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)
print(f"\n=== Training {EPOCHS} epoch ===")
print(f"⏱  CPU training is slow — expect ~30-45 minutes for one epoch. Make coffee.\n")

def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return loss_sum / len(loader), 100. * correct / total

best_test_acc = 0.0


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0                                                                                                                                      
    running_correct = 0
    running_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

          # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)                                                                                                                   

          # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

          # Running stats                                                                                                                                     
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        running_total += labels.size(0)
        running_correct += predicted.eq(labels).sum().item()

          # Live display in progress bar
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * running_correct / running_total
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"})

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * running_correct / running_total

      # Step LR scheduler
    scheduler.step()

      # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f},train_acc={epoch_acc:.2f}%, "f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%")

      # Save best checkpoint based on test accuracy
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
            "class_names": class_names,
        }, "checkpoints/resnet18_best.pt")
        print(f"  ✅ New best test accuracy: {test_acc:.2f}% — savedcheckpoint")

print(f"\n🎯 Training complete. Best test accuracy: {best_test_acc:.2f}%")
print(f"📁 Best checkpoint: checkpoints/resnet18_best.pt")

