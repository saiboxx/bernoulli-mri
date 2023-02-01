from typing import Optional

import torch
from torch import Tensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def mse(gt: Tensor, pred: Tensor) -> float:
    """Compute Mean Squared Error (MSE)"""
    return float(torch.mean((gt - pred) ** 2))


def nmse(gt: Tensor, pred: Tensor) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return float(torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2)


def psnr(
    gt: Tensor, pred: Tensor, maxval: Optional[float] = None
) -> float:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()

    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=maxval)
    return float(psnr)



def ssim(
    gt: Tensor, pred: Tensor, maxval: Optional[float] = None
) -> float:
    """Compute Structural Similarity Index Metric (SSIM)"""
    maxval = gt.max() if maxval is None else maxval

    gt_np = gt.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()
    ssim = structural_similarity(gt_np, pred_np, data_range=maxval)
    return float(ssim)

METRIC_FUNCS = {
    'MSE': mse,
    'NMSE': nmse,
    'PSNR': psnr,
    'SSIM': ssim,
}