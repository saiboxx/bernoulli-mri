import torch

from src.run import run_dataset_optim

BASE_CONFIG = {
    # Dataset ID, currently implemented: knee, brain, acdc
    'dataset': 'knee',
    # Root location of the dataset
    'dataset_root': 'data',
    # Number of samples per step
    'batch_size': 32,
    # Number of ProM optimization steps
    'steps': 2500,
    # Whether the target should be the segmentation task
    'use_seg': False,
    # Learning rate
    'learning_rate': 1e-2,
    # Number of samples from Bernoulli distribution
    'bern_samples': 4,
    # Mask shape, 'f' implies full 2D mask
    'mask_style': 'f',
    # Workers for dataloader
    'num_workers': 16,
    # Amounts to acceleration factor
    'dense_target': 1 / 8,
    # Exploration phase
    'dense_start': 0.10,
    # Exploitation phase
    'dense_end': 0.90,
    # Device for training
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # Logging directory
    'log_dir': 'logs/knee',
    # Number of logged images
    'log_imgs': 10,
    # Logging interval
    'log_interval': 100,
    # Seed
    'seed': None,
    # Number of members in case of model averaging
    'ensemble_members': None
}

if __name__ == '__main__':
    run_dataset_optim(BASE_CONFIG)