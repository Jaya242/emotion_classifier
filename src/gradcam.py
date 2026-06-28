"""Generate GradCAM heatmaps from the best checkpoint."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models, transforms
from PIL import Image
import io
from datasets import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import random
import os

CHECKPOINT = "checkpoints/resnet18_best.pt"

# 1. Load checkpoint
ckpt = torch.load(CHECKPOINT, map_location="cpu")
class_names = ckpt["class_names"]
print(f"Loaded checkpoint with test_acc={ckpt.get('test_acc', 
  'unknown')}")
print(f"Classes: {class_names}")

# 2. Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Set up GradCAM on the last conv layer
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 4. Load dataset
dataset = load_dataset("Jeneral/fer-2013")
test = dataset["test"]

# 5. Same transforms as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
])

def decode_img(sample):
    data = sample["img_bytes"]
    return Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data

# 6. Find one good sample per emotion class
samples_per_class = {}
random.seed(42)
indices = random.sample(range(len(test)), len(test))
for idx in indices:
    sample = test[idx]
    label = sample["labels"]
    if label not in samples_per_class:
        samples_per_class[label] = sample
    if len(samples_per_class) == len(class_names):
        break

  # 7. Generate GradCAM for each
os.makedirs("gradcam_samples", exist_ok=True)
fig, axes = plt.subplots(2, 7, figsize=(20, 6))
for i, (class_idx, sample) in enumerate(sorted(samples_per_class.items())):
    img = decode_img(sample).convert("RGB").resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    rgb_img = np.array(img) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam,use_rgb=True)

    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Original: {class_names[class_idx]}")
    axes[0, i].axis("off")
    axes[1, i].imshow(cam_image)
    axes[1, i].set_title("GradCAM")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("gradcam_samples/all_classes.png", dpi=100)
plt.close()
print("✅ Saved gradcam_samples/all_classes.png")