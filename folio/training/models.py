"""ConvNeXt-V2 heads. Torch + timm imported lazily.

Signatures are chosen to match the inference wrappers exactly:
  * OrientationNet.forward(x)        -> logits[B,4]      (OrientationClassifier)
  * FolioCountNet.forward(x, aux)    -> logits[B,3]      (FolioCountClassifier,
    aux = [aspect_ratio, gutter_valley_ratio] from geometric_priors)
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover
    timm = None

_BACKBONE = "convnextv2_tiny.fcmae_ft_in22k_in1k"


def _backbone(pretrained: bool):
    if timm is None:
        raise RuntimeError("pip install timm to train")
    net = timm.create_model(_BACKBONE, pretrained=pretrained, num_classes=0)
    return net, net.num_features


class OrientationNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, f = _backbone(pretrained)
        self.head = nn.Linear(f, 4)

    def forward(self, x):
        return self.head(self.backbone(x))


class BlankNet(nn.Module):
    """content vs blank/non-content -> logits[B,2] (matches BlankClassifier)."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone, f = _backbone(pretrained)
        self.head = nn.Linear(f, 2)

    def forward(self, x):
        return self.head(self.backbone(x))


class FolioCountNet(nn.Module):
    def __init__(self, pretrained: bool = True, n_aux: int = 2):
        super().__init__()
        self.backbone, f = _backbone(pretrained)
        self.aux = nn.Sequential(nn.Linear(n_aux, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(f + 32, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 3),
        )

    def forward(self, x, aux):
        feat = self.backbone(x)
        return self.head(torch.cat([feat, self.aux(aux)], dim=1))


def export_torchscript(model: nn.Module, task: str, size: int, path: str,
                       device: str = "cpu"):
    """Trace to TorchScript so it loads via torch.jit.load in the wrappers."""
    model.eval().to(device)
    ex_img = torch.randn(1, 3, size, size, device=device)
    if task in ("orientation", "blank"):
        scripted = torch.jit.trace(model, ex_img)
    else:
        ex_aux = torch.randn(1, 2, device=device)
        scripted = torch.jit.trace(model, (ex_img, ex_aux))
    scripted.save(path)
    return path
