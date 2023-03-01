from typing import Optional

import torch
from torch import Tensor
from piqa.ssim import ssim
from piqa.psnr import psnr
from piqa.utils.functional import gaussian_kernel


def mse(gt: Tensor, pred: Tensor) -> float:
    """Compute Mean Squared Error (MSE)"""
    return float(torch.mean((gt - pred) ** 2))


def nmse(gt: Tensor, pred: Tensor) -> float:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return float(torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2)


class MetricAgent:
    def __init__(self):
        self.psnrs = []
        self.ssims = []
        self.nmses = []

    def reset(self):
        self.psnrs.clear()
        self.ssims.clear()
        self.nmses.clear()

    def aggregate(self):
        r = {
            'PSNR': sum(self.psnrs) / len(self.psnrs),
            'SSIM': sum(self.ssims) / len(self.ssims),
            'NMSE': sum(self.nmses) / len(self.nmses),
        }
        self.reset()
        return r

    def __call__(self, prediction: Tensor, target: Tensor) -> None:
        assert len(prediction) == len(target)

        # Normalize values
        for i in range(len(prediction)):
            min_val = torch.min(target[i])
            max_val = torch.max(target[i])

            target[i] = (target[i] - min_val) / (max_val - min_val)
            prediction[i] = (prediction[i] - min_val) / (max_val - min_val)
        prediction.clamp_(0, 1)

        # Compute metrics
        psnrs = psnr(prediction, target)
        self.psnrs.extend(psnrs.tolist())

        kernel = gaussian_kernel(7, sigma=1.).repeat(prediction.shape[1], 1, 1).to(
            prediction.device)
        ssims = ssim(prediction, target, kernel)[0]
        self.ssims.extend(ssims.tolist())

        nmses = [nmse(t, p) for p, t in zip(prediction, target)]
        self.nmses.extend(nmses)
