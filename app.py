"""Gradio webcam emotion classifier with GradCAM toggle."""
import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

CHECKPOINT = "checkpoints/resnet18_best.pt"

ckpt = torch.load(CHECKPOINT, map_location="cpu")
class_names = ckpt["class_names"]
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224, 0.225]),
])


def classify(image, show_gradcam):
    if image is None:
        return "Upload or capture an image first.", None

    img = image.convert("RGB").resize((224, 224))
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_idx = probs.argmax().item()
    confidence = probs[top_idx].item() * 100
    label_text = f"### Prediction: **{class_names[top_idx]}**({confidence:.1f}%)\n\n"
    label_text += "All scores:\n"
    for i, name in enumerate(class_names):
        label_text += f"- {name}: {probs[i].item()*100:.1f}%\n"

    if show_gradcam:
        grayscale_cam = cam(input_tensor=input_tensor,targets=None)[0]
        rgb_img = np.array(img) / 255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam,use_rgb=True)
        return label_text, Image.fromarray(cam_image)

    return label_text, img

demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Upload or webcam capture",sources=["upload", "webcam"]),
        gr.Checkbox(label="Show GradCAM heatmap", value=True),
    ],
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Image(type="pil", label="Image (with GradCAM if toggled)"),
    ],
    title="🎭 Facial Emotion Classifier + GradCAM",
    description="ResNet18 fine-tuned on FER2013. Upload a face image or use webcam to see the predicted emotion + GradCAM heatmap showing which face regions drove the prediction.",
    article="Built as part of an AI/ML portfolio sprint. [GitHub](https://github.com/Jaya242/emotion_classifier)",
    flagging_mode="never",
)
  
                       
if __name__ == "__main__":
    demo.launch()