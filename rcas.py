"""Local center recomposition and radial-constrained anomaly synthesis."""
import torch
import torch.nn.functional as F


def recompose_centers(features, center_bank, batch_size):
    """Match each patch to the most cosine-similar normal center vector."""
    patches = features.shape[0] // batch_size
    matched = torch.zeros_like(features)
    for i in range(batch_size):
        block = features[i * patches:(i + 1) * patches]
        similarity = torch.mm(F.normalize(block, dim=1), F.normalize(center_bank, dim=1).T)
        indices = torch.argmax(similarity, dim=1)
        matched[i * patches:(i + 1) * patches] = center_bank[indices]
    return matched


def synthesize_anomalies(features, centers, sigma, clamp_min=0.1, clamp_max=3.0):
    """Reverse inward noise and apply bounded distance-guided scaling."""
    eps = 1e-6
    radial = features - centers
    distance = torch.norm(radial, dim=1, keepdim=True)
    radial_direction = radial / (distance + eps)
    noise = torch.normal(0, sigma, features.shape).to(features.device)
    magnitude = torch.norm(noise, dim=1, keepdim=True)
    direction = noise / (magnitude + eps)
    cosine = torch.sum(direction * radial_direction, dim=1, keepdim=True)
    direction = torch.where(cosine < 0, -direction, direction)
    scale = magnitude / (distance + eps)
    if clamp_min is not None and clamp_max is not None:
        scale = torch.clamp(scale, min=clamp_min, max=clamp_max)
    return features + (scale * magnitude * direction).detach()
