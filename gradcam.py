import torch
import torch.nn as nn
import cv2
import numpy as np

def generate_gradcam(model, input_tensor, class_idx):
    """
    Generate Grad-CAM heatmap overlay for given model and input tensor.
    Returns numpy BGR image (overlayed on the original input).
    """
    model.eval()

    # Ensure gradients are enabled
    input_tensor.requires_grad = True

    # Find target convolution layer (last Conv2d)
    target_layer = None
    if hasattr(model, 'base') and hasattr(model.base, 'features'):
        for layer in reversed(model.base.features):
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
            if target_layer:
                break
    else:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break

    if target_layer is None:
        raise ValueError("No Conv2d layer found for Grad-CAM computation.")

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        score = output[0, class_idx]
        model.zero_grad()
        score.backward(retain_graph=True)

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM failed to capture gradients or activations.")

        grads = gradients[0].mean(dim=(2, 3), keepdim=True)
        acts = activations[0]
        cam = torch.relu((grads * acts).sum(dim=1)).squeeze().detach().cpu().numpy()

        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # Convert tensor to original image
        img_np = input_tensor.squeeze().detach().cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = np.uint8(255 * img_np)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Overlay Grad-CAM on original image
        overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

    finally:
        fh.remove()
        bh.remove()

    return overlay
