"""VICReg (sample dimension is the batch) and masked reflectance Huber."""
import torch
import torch.nn.functional as F


def vicreg(z1, z2, *, eps=1e-4):
    """Returns invariant, variance and covariance loss terms (unweighted).

    Batch >= 2 is required for variance/covariance estimates; batch 16+
    is recommended. All terms evaluated in float32 under BF16 autocast.
    """
    z1, z2 = z1.float(), z2.float()
    if z1.ndim != 2 or z1.shape != z2.shape or z1.shape[0] < 2:
        raise ValueError('VICReg requires matching [B,D] tensors with B >= 2')
    inv = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0, unbiased=True) + eps)
    std2 = torch.sqrt(z2.var(dim=0, unbiased=True) + eps)
    var = (F.relu(1 - std1).mean() + F.relu(1 - std2).mean()) * 0.5
    centered = [z - z.mean(0) for z in (z1, z2)]
    cov = z1.new_zeros(())
    for x in centered:
        c = (x.T @ x) / (x.shape[0] - 1)
        cov += (c.square().sum() - c.diagonal().square().sum()) / z1.shape[1]
    return inv, var, cov * 0.5


def masked_huber(model, image, valid, output, delta=1.0):
    """Calculate Huber ONLY for valid pixels in masked 8x8 scoring patches."""
    prediction = output['predicted_patches'].float()
    target_image = torch.where(valid.expand_as(image).bool(), image, 0.).float()
    target = model.patchify(target_image)
    valid_patch = model.patchify(valid.expand_as(image).float()) > 0.5
    scored = output['patch_mask'].unsqueeze(-1) & valid_patch
    if not scored.any():
        raise RuntimeError('No valid masked pixels in batch')
    return F.huber_loss(prediction[scored], target[scored], delta=float(delta))
