"""A small video model of the flow: next vorticity frame from the current frame.

Fully convolutional with circular padding (the box is periodic). Dilated residual blocks give a receptive field that
covers the whole 64 x 64 box, which the model needs: velocity depends on vorticity everywhere (Biot-Savart).
Sites for interpretability: 0 = after the stem, i = after residual block i. Each site is [B, C, H, W].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.d = d
        self.conv = nn.Conv2d(c, c, 3, dilation=d)
        self.mix = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return x + self.mix(F.gelu(self.conv(F.pad(x, (self.d,) * 4, mode="circular"))))


class FluidNet(nn.Module):
    def __init__(self, width=32, dilations=(1, 2, 4, 8, 16, 1)):
        super().__init__()
        self.width, self.dilations = width, tuple(dilations)
        self.stem = nn.Conv2d(1, width, 3)
        self.blocks = nn.ModuleList([Block(width, d) for d in dilations])
        self.head = nn.Conv2d(width, 1, 1)
        self.register_buffer("in_scale", torch.tensor(1.0))
        self.register_buffer("out_scale", torch.tensor(1.0))

    @property
    def n_sites(self):
        return len(self.blocks) + 1

    def forward(self, w, return_acts=False, edit=None):
        """w: [B, H, W] vorticity. Returns the predicted next frame [B, H, W] (and the site activations).
        edit(site, h) -> h lets a caller patch or steer a site in the forward pass."""
        x = (w / self.in_scale).unsqueeze(1)
        h = self.stem(F.pad(x, (1, 1, 1, 1), mode="circular"))
        if edit is not None:
            h = edit(0, h)
        acts = [h]
        for i, b in enumerate(self.blocks):
            h = b(h)
            if edit is not None:
                h = edit(i + 1, h)
            acts.append(h)
        out = w + self.head(h).squeeze(1) * self.out_scale
        return (out, acts) if return_acts else out


def save(model, path, extra=None):
    torch.save({"state": model.state_dict(), "width": model.width, "dilations": model.dilations,
                "extra": extra or {}}, path)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = FluidNet(ck["width"], ck["dilations"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck.get("extra", {})
