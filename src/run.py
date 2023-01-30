import os
from typing import Dict

import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from src.constraint import ScoreConstrainer
from src.dense_rate import DenseRateScheduler
from src.optimization import get_mask_handler
from src.utils import (
    get_temperature,
    ifft2c,
    plot_heatmap,
    min_max_normalize,
    MaskLogger
)
from src.datasets import ACDCDataset, BrainDataset, KneeDataset


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def run_dataset_optim(cfg: Dict) -> None:

    # Initialize objects
    # ----------------------------------------------------------------------------------
    torch.manual_seed(cfg['seed'])

    if cfg['use_seg']:
        loss_func = nn.MSELoss(reduction='none')
    else:
        loss_func = nn.MSELoss()

    if cfg['dataset'] == 'acdc':
        ds = ACDCDataset(os.path.join(cfg['dataset_root'], 'ACDC'), train=True)
    elif cfg['dataset'] == 'brain':
        ds = BrainDataset(os.path.join(cfg['dataset_root'], 'Task01_BrainTumour'), train=True)
    elif cfg['dataset'] == 'knee':
        ds = KneeDataset(os.path.join(cfg['dataset_root'], 'knee_fastmri'), train=True)
    else:
        raise ValueError('Dataset {} unknown.'.format(cfg['dataset']))

    data_loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=16)
    data_iter = iter(cycle(data_loader))

    dense_scheduler = DenseRateScheduler(
            target=cfg['dense_target'],
            start_epoch=int(cfg['dense_start'] * cfg['steps']),
            stop_epoch=int(cfg['dense_end'] * cfg['steps']),
        )

    # Create container for masks during training
    logger = MaskLogger(cfg['log_dir'])

    # Create mask handler that also contains scores
    mask_handler = get_mask_handler(
        name=cfg['mask_style'],
        height=320 if cfg['dataset'] == 'knee' else 256,
        width=320 if cfg['dataset'] == 'knee' else 256,
        device=cfg['device'],
    )

    # Initialize optimizer with mask scores
    optimizer = Adam([mask_handler.get_scores()], lr=cfg['learning_rate'])

    # Create constrainer for projection of scores to dense_rate
    constrainer = ScoreConstrainer(mask_handler.get_scores())

    log_recs = []
    log_masks = []

    for step in (pbar := tqdm(range(1, cfg['steps'] + 1))):

        samples = next(data_iter)
        img = samples['img']
        img_k = samples['k_space']

        img = img.to(cfg['device'])
        img_k = img_k.to(cfg['device'])

        img_k_batch = torch.repeat_interleave(img_k, repeats=cfg['bern_samples'], dim=0)
        img_batch = torch.repeat_interleave(img, repeats=cfg['bern_samples'], dim=0)

        # Get temperature for categorical "softness"
        temperature = get_temperature(step, cfg['steps'])

        # Map scores to valid probability space
        dense_rate = dense_scheduler.get_dense_rate()
        constrainer.constrain(dense_rate=dense_rate)
        dense_scheduler.advance()

        # Sample from distribution
        mask = mask_handler.sample_mask(
            temperature=temperature, num_samples=cfg['bern_samples'] * img.shape[0]
        )

        # Compute image with mask
        img_pred = ifft2c(img_k_batch * mask + 0.0)
        img_mag = torch.abs(img_pred)

        # Compute loss between full and undersampled image
        loss = loss_func(img_mag, img_batch)

        if cfg['use_seg']:
            seg = samples['seg']
            seg[seg != 0] = cfg['seg_weight']
            seg[seg == 0] = 1
            seg = torch.repeat_interleave(
                seg, repeats=cfg['bern_samples'], dim=0).to(cfg['device']
                                                            )
            loss *= seg
            loss = torch.mean(loss)

        # Optimize scores
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            pbar.set_description(
                'L: {:.2E} | D: {:.3f}'.format(
                    float(loss), float(torch.mean(mask_handler.get_scores()))
                )
            )

        if step == 1 or step % cfg['log_interval'] == 0:
            logger.append(probs=mask_handler.get_mask_distribution(),
                          step=step, dense_rate=dense_scheduler.get_dense_rate())

        if cfg['log_imgs'] > 0 and step % (cfg['steps'] // cfg['log_imgs']) == 0:
            img_rec = torch.abs(img_pred[0])

            img_low_q = torch.quantile(img, 0.01)
            img_high_q = torch.quantile(img, 0.99)

            img_rec = min_max_normalize(img_rec, img_low_q, img_high_q)

            log_recs.append(img_rec.detach().cpu())
            log_masks.append(mask[0].detach().cpu())

    # ------------------------------------------------------------------------------
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger.write()

    # Construct training progress grid
    img_org = min_max_normalize(img[0], img_low_q, img_high_q)
    log_recs.append(img_org.cpu())
    log_masks.append(torch.zeros_like(img[0]).cpu())

    save_image(
        log_recs + log_masks,
        fp=cfg['log_dir'] + '/progress.png',
        nrow=len(log_recs),
    )

    # Construct heatmap
    plot_heatmap(
        mask_handler.get_mask_distribution(),
        save_path=cfg['log_dir'] + '/heatmap.png',
    )

    # Construct histogram of scores
    plt.hist(mask_handler.get_scores().detach().cpu().flatten(), bins=20)
    plt.savefig(cfg['log_dir'] + '/histogram.png')