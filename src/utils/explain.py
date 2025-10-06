import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms.functional import to_pil_image

def gradcam(model, image, label, device, layer_name='conv3'):
    """
    Visualize Grad-CAM for a given image and model layer.
    Displays original image and Grad-CAM heatmap overlay side by side.
    """
    model.eval()
    image = image.unsqueeze(0).to(device)

    features, grads = {}, {}

    def forward_hook(module, inp, out):
        features['value'] = out

    def backward_hook(module, grad_in, grad_out):
        grads['value'] = grad_out[0]

    # Register hooks
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    # Forward + backward
    output = model(image)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, pred_class].backward()

    # Compute Grad-CAM
    gradient = grads['value'].mean(dim=[2, 3], keepdim=True)
    cam = (gradient * features['value']).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam / cam.max()

    # Convert to NumPy and resize to match input image size
    cam_resized = cv2.resize(cam.detach().cpu().numpy(), (image.shape[3], image.shape[2]))

    # Convert image for display
    img_np = image.squeeze().detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Create overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.9 * heatmap / 255.0 + np.repeat(img_np[..., None], 3, axis=2)  # overlay on grayscale

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f"Original (Label: {label})")
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM → Pred: {pred_class}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def show_misclassified_gradcams(model, dataloader, device, class_names, max_samples=10, layer_name="conv3"):
    """
    Show Grad-CAM visualizations for misclassified samples.
    Displays up to `max_samples` pairs (Original vs Grad-CAM).
    """
    model.eval()
    misclassified = []

    # --- Find misclassified samples ---
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), labels[i].item(), preds[i].item()))
                    if len(misclassified) >= max_samples:
                        break
            if len(misclassified) >= max_samples:
                break

    print(f"Found {len(misclassified)} misclassified samples.")

    # --- Plot Grad-CAM for each ---
    n = len(misclassified)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))

    for idx, (img, true_label, pred_label) in enumerate(misclassified):
        # Compute Grad-CAM heatmap
        image = img.unsqueeze(0).to(device)
        features, grads = {}, {}

        def forward_hook(module, inp, out):
            features['value'] = out

        def backward_hook(module, grad_in, grad_out):
            grads['value'] = grad_out[0]

        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

        output = model(image)
        pred_class = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, pred_class].backward()

        gradient = grads['value'].mean(dim=[2, 3], keepdim=True)
        cam = (gradient * features['value']).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / (cam.max()+1e-8)

        # Prepare visualizations
        cam_resized = cv2.resize(cam.detach().cpu().numpy(), (image.shape[3], image.shape[2]))
        img_np = image.squeeze().detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = 0.5 * heatmap / 255.0 + np.repeat(img_np[..., None], 3, axis=2)
        overlay = np.clip(overlay, 0, 1)


        # Plot
        ax1, ax2 = axes[idx]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f"True: {class_names[true_label]} | Pred: {class_names[pred_label]}")
        ax1.axis('off')

        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM → Pred: {class_names[pred_label]}")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

