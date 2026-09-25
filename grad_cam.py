"""
Grad-CAM (Gradient-weighted Class Activation Mapping) für das
Jubaea-Klassifikationsmodell.

Zeigt, welche Bildregionen das Modell für seine Vorhersage
verwendet hat, als Heatmap über dem Originalbild.

Benötigt kein erneutes Training, nur das bereits trainierte Modell.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """
    Registriert Hooks auf einer Zielschicht (bei ResNet18 typischerweise
    net.layer4[-1], die letzte Convolutional-Schicht) und berechnet daraus
    die Klassenaktivierungskarte.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: bereits transformiertes Bild, Shape [1, 3, H, W]
        target_class: Klassenindex, für den die Heatmap berechnet wird.
                      None = die vom Modell vorhergesagte Klasse.

        Rückgabe: (cam als 2D-numpy-Array in [0, 1], target_class, confidence)
        """
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0]      # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Wichtigkeit jedes Feature-Kanals = Mittelwert seiner Gradienten
        weights = gradients.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Nur positive Beiträge interessieren uns
        cam = F.relu(cam)

        # Auf [0, 1] normalisieren
        cam = cam / (cam.max() + 1e-8)

        confidence = torch.softmax(output, dim=1)[0, target_class].item()

        return cam.cpu().numpy(), target_class, confidence


def overlay_heatmap(image_path, cam, image_size=224, alpha=0.5):
    """
    Legt die (kleine) Grad-CAM-Karte als farbige Heatmap über
    das auf image_size skalierte Originalbild.

    Rückgabe: (original_bild, heatmap, overlay), jeweils als
    numpy-Array mit Werten in [0, 1], geeignet für plt.imshow.
    """
    image = Image.open(image_path).convert("RGB").resize(
        (image_size, image_size)
    )
    image_np = np.array(image) / 255.0

    # cam ist kleiner als das Bild (z.B. 7x7 bei ResNet18) -> hochskalieren
    heatmap = Image.fromarray(np.uint8(cam * 255)).resize(
        (image_size, image_size),
        resample=Image.BILINEAR
    )
    heatmap = np.array(heatmap) / 255.0

    colored_heatmap = cm.jet(heatmap)[:, :, :3]

    overlay = (1 - alpha) * image_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return image_np, colored_heatmap, overlay


def show_gradcam(model, image_path, transform, device, classes, save_path=None):
    """
    Komplettablauf für ein einzelnes Bild: lädt es, berechnet Grad-CAM,
    zeigt Original / Heatmap / Overlay nebeneinander.

    Beispiel (in deinem bestehenden Notebook, nach dem Training):

        target_layer = net.layer4[-1]
        grad_cam = GradCAM(net, target_layer)

        show_gradcam(
            model=grad_cam,
            image_path="Data/Jubaea/beispiel.jpg",
            transform=test_transform,
            device=device,
            classes=["Not_Jubaea", "Jubaea"]
        )
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    cam, predicted_class, confidence = model.generate(input_tensor)

    original, heatmap, overlay = overlay_heatmap(image_path, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay\n{classes[predicted_class]} ({confidence:.1%})"
    )
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()