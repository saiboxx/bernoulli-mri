import torch
from monai.losses import DiceCELoss

from src.losses import SegmentationProxyLoss
from src.run import run_dataset_optim
from src.utils import get_unet

BASE_CONFIG = {
    'dataset': 'acdc',
    'dataset_root': 'data',
    'batch_size': 32,
    'steps': 2500,
    'use_seg': False,
    'learning_rate': 1e-2,
    'bern_samples': 4,
    'mask_style': 'f',
    'num_workers': 16,
    'dense_target': 1 / 8,
    'dense_start': 0.10,
    'dense_end': 0.90,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'log_dir': 'logs/acdc',
    'log_imgs': 10,
    'log_interval': 100,
    'seed': None,
    'ensemble_members': 10
}

if __name__ == '__main__':
    # Load U-net segmentation model
    model = get_unet('acdc')

    # Create custom segmentation loss function
    loss_func = DiceCELoss(
        softmax=True,
        include_background=False,
        to_onehot_y=True
    )
    loss_func = SegmentationProxyLoss(model=model, seg_loss_func=loss_func).to(BASE_CONFIG['device'])

    # Supply loss function to optimization routine
    run_dataset_optim(BASE_CONFIG, loss_func=loss_func)