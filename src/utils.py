import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
from monai.networks import one_hot
import numpy as np
import torch
from torch import Tensor
from torch.fft import ifftshift, fftshift, ifft2, fft2
from torchvision.utils import save_image



def ifft2c(data: Tensor, norm: str = 'forward') -> Tensor:
    data = ifftshift(data)
    data = ifft2(data, norm=norm)
    data = fftshift(data)
    return data


def fft2c(data: Tensor, norm: str = 'forward') -> Tensor:
    data = ifftshift(data)
    data = fft2(data, norm=norm)
    data = fftshift(data)
    return data


def get_temperature(current_epoch: int, max_epoch: int) -> float:
    return (1 - 0.03) * (1 - current_epoch / max_epoch) + 0.03


def load_h5_slice(fp: str, slice_idx: int, coil_idx: Optional[int] = None) -> Tensor:
    with h5py.File(fp, 'r') as f:
        arr_k = f['kspace'][slice_idx]

    if coil_idx is not None:
        arr_k = arr_k[coil_idx]

    return torch.from_numpy(arr_k).unsqueeze(0)


def convert_complex_img_to_real(img: Tensor):
    img = torch.abs(img)
    return (img - img.min()) / (img.max() - img.min())


def normalize(
    x: Tensor, mean: Optional[Tensor] = None, std: Optional[Tensor] = None
) -> Tensor:
    if mean is None:
        mean = torch.mean(x)
    if std is None:
        std = torch.std(x)
    return (x - mean) / std


def reverse_normalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return x * std + mean


def min_max_normalize(
    x: Tensor,
    min_val: Optional[Tensor] = None,
    max_val: Optional[Tensor] = None,
    min_quantile: float = 0.0,
    max_quantile: float = 1.0,
) -> Tensor:

    if min_val is None:
        min_val = torch.quantile(x, min_quantile)

    if max_val is None:
        max_val = torch.quantile(x, max_quantile)

    x = torch.clamp(x, min_val, max_val)
    return (x - min_val) / (max_val - min_val)


def plot_image(img: Tensor, cmap: str = 'gray') -> None:
    if torch.is_complex(img):
        img = convert_complex_img_to_real(img)

    plt.imshow(img.detach().cpu().permute(1, 2, 0), cmap=cmap)
    plt.axis('off')
    plt.show()


def plot_heatmap(
    scores: Tensor,
    cmap: str = 'magma',
    show: bool = False,
    save_path: Optional[str] = None,
) -> None:
    h = scores.squeeze().detach().cpu().clamp(0, 1)
    plt.figure(figsize=(15, 15))
    plt.imshow(h, cmap=cmap, vmin=0, vmax=1)
    plt.axis('off')

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


def save_img(img: Tensor, fp: str, **kwargs) -> None:
    if torch.is_complex(img):
        img = convert_complex_img_to_real(img)
    save_image(tensor=img, fp=fp, **kwargs)


class MaskLogger:
    def __init__(self, log_dir: str) -> None:
        self.probs = []
        self.steps = []
        self.dense_rates = []
        self.log_dir = log_dir

    def append(self, probs: Tensor, step: int, dense_rate: float) -> None:
        self.probs.append(probs.detach().cpu())
        self.steps.append(step)
        self.dense_rates.append(dense_rate)

    def write(self, f_name: str = 'results.pt') -> None:
        content = {
            'scores': self.probs,
            'steps': self.steps,
            'dense_rates': self.dense_rates
        }
        torch.save(obj=content, f=os.path.join(self.log_dir, f_name))


def convert_brats_labels(seg):
    pred = one_hot(seg, num_classes=4)

    res = torch.zeros_like(pred)

    res[:, 0] = pred[:, 0]
    res[:, 1] = torch.where(torch.sum(pred[:, 1:], dim=1) > 0, 1, 0)
    res[:, 2] = torch.where(torch.sum(pred[:, 2:], dim=1) > 0, 1, 0)
    res[:, 3] = pred[:, 3]
    return res


def get_top_k_mask(scores: Tensor, acc_fac: int) -> Tensor:
    assert len(scores.shape) == 2, 'Expected tensor to be (H x W)'

    num_elem = scores.numel()
    k = int(num_elem / acc_fac)

    v, i = torch.topk(scores.flatten(), k)
    idxs = torch.tensor(np.asarray(np.unravel_index(i.cpu().numpy(), scores.shape)).T)

    mask = torch.zeros_like(scores)
    mask[idxs[:, 0], idxs[:, 1]] = 1
    return mask
