"""ANNAlyzer adaptation of Flownomaly with a separately fine-tuned ViT."""

import math

import torch
from torch import nn
from torch.nn import functional as F


ENCODER = "vit_small_patch14_dinov2.lvd142m"
LAYERS = (7, 9, 11)
PATCH = 14
PAD = 3
TILE = 512
PADDED = TILE + 2 * PAD
GRID = PADDED // PATCH
DIMENSION = 384 * len(LAYERS)


def load_encoder(weights):
    from safetensors.torch import load_file
    from timm import create_model

    model = create_model(ENCODER, pretrained=False, img_size=PADDED, num_classes=0)
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.requires_grad_(False).eval()


def tokens(encoder, padded):
    features = encoder.forward_intermediates(padded, indices=LAYERS, norm=True,
                                             output_fmt="NLC", intermediates_only=True)
    return torch.cat(features, dim=-1).float()


def enable_encoder_training(encoder):
    encoder.blocks[-2:].requires_grad_(True)
    encoder.norm.requires_grad_(True)


def masked_token_loss(student, teacher, padded, indices):
    grid_mask = torch.zeros(GRID * GRID, dtype=torch.bool, device=padded.device)
    grid_mask[indices] = True
    pixel_mask = grid_mask.view(GRID, GRID).repeat_interleave(PATCH, 0).repeat_interleave(PATCH, 1)
    with torch.no_grad():
        target = tokens(teacher, padded)[0, indices, -384:]
    prediction = tokens(student, padded.masked_fill(pixel_mask, 0))[0, indices, -384:]
    return F.mse_loss(prediction, target)


class VelocityField(nn.Module):
    def __init__(self, dimension=DIMENSION, width=256):
        super().__init__()
        self.input = nn.Linear(dimension + 9, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                                  nn.Linear(width, width)) for _ in range(3))
        self.output = nn.Linear(width, dimension)

    def forward(self, points, time):
        frequencies = points.new_tensor((1, 2, 4, 8)) * math.pi
        angles = time * frequencies
        features = torch.cat((points, time, angles.sin(), angles.cos()), dim=-1)
        features = self.input(features)
        for block in self.blocks:
            features = features + block(features)
        return self.output(features)


def field_loss(field, normal, generator):
    noise = torch.randn(normal.shape, device=normal.device, generator=generator)
    time = torch.rand((len(normal), 1), device=normal.device, generator=generator)
    mixed = time * normal + (1 - time) * noise
    return F.mse_loss(field(mixed, time), normal - noise)


def residual(field, query, noise, time):
    time_column = query.new_full((len(query), 1), time)
    mixed = time * query + (1 - time) * noise
    recovered = mixed + (1 - time) * field(mixed, time_column)
    return query - recovered


@torch.no_grad()
def operating_time(field, normal_views, seed, candidates=tuple(step / 20 for step in range(1, 20))):
    """Estimate the residual Jacobian gain and divide by the normal residual tail."""
    scores = {}
    for time in candidates:
        by_view = []
        for index, query in enumerate(normal_views):
            generator = torch.Generator(device=query.device).manual_seed(seed + index)
            repetitions = []
            for _ in range(4):
                noise = torch.randn(query.shape, device=query.device, generator=generator)
                gains = []
                for _ in range(4):
                    probe = torch.randint(0, 2, query.shape, device=query.device, generator=generator)
                    probe = (2 * probe.float() - 1) / math.sqrt(query.shape[-1])
                    error, direction = torch.func.jvp(lambda x: residual(field, x, noise, time), (query,), (probe,))
                    gains.append(direction.square().sum(dim=-1).mean())
                tail = error.square().sum(dim=-1).topk(max(1, math.ceil(len(query) * 0.01))).values.mean()
                repetitions.append(torch.stack(gains).mean() / (tail + 1e-8))
            by_view.append(float(torch.stack(repetitions).quantile(0.5)))
        scores[time] = float(torch.tensor(by_view, dtype=torch.float64).quantile(0.5))
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite normal-only calibration score")
    return max(scores, key=scores.get), scores


def tile_prediction(field, features, center, spread, time, generator):
    query = (features[0] - center) / spread
    noise = torch.randn(query.shape, device=query.device, generator=generator)
    energy = residual(field, query, noise, time).square().sum(dim=-1).reshape(1, 1, GRID, GRID)
    plain = F.interpolate(energy, (PADDED, PADDED), mode="bilinear", align_corners=False)
    core = plain[0, 0, PAD:-PAD, PAD:-PAD]
    image_score = core.flatten().topk(math.ceil(TILE * TILE * 0.01)).values.mean().item()
    axis = torch.arange(-3, 4, dtype=energy.dtype, device=energy.device)
    gaussian = torch.exp(-0.5 * (axis / 0.8).square())
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian)[None, None]
    padded = F.pad(energy, (3, 3, 3, 3), mode="reflect")
    smoothed = F.conv2d(padded, kernel)
    upsampled = F.interpolate(smoothed, (PADDED, PADDED), mode="bilinear", align_corners=False)
    return upsampled[0, 0, PAD:-PAD, PAD:-PAD], image_score
