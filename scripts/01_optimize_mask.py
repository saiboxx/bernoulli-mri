from typing import Dict

import torch
from torch.nn import MSELoss

from src.config import get_configuration
from src.optimization import MaskOptimizer
from src.losses import SobelLoss, SobelMSELoss

BASE_CONFIG = {
    'file_path': 'data/file1001948.h5',
    'slice_idx': 17,
    'coil_idx': None,
    'cropping': (320, 320),
    'steps': 5000,
    'learning_rate': 1e-2,
    'bern_samples': 16,
    'mask_style': 'f',
    'dense_target': 1 / 64,
    'dense_start': 0.10,
    'dense_end': 0.85,
    'device': torch.device('cuda'),
    'log_dir': 'logs/knee_full',
    'log_imgs': 10,
}


def run(cfg: Dict) -> None:
    cfg_obj = get_configuration(cfg)
    mask_optimizer = MaskOptimizer(cfg_obj)

    #loss_func = SobelMSELoss(alpha=0.9).cuda()
    loss_func = MSELoss()
    mask_optimizer.run(loss_func=loss_func)


if __name__ == '__main__':
    run(BASE_CONFIG)
