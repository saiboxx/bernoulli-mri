import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

from src.constraint import ScoreConstrainer
from src.dense_rate import DenseRateScheduler
from src.optimization import get_mask_handler
from src.utils import (
    get_temperature,
    ifft2c,
    plot_heatmap,
    min_max_normalize,
    MaskLogger,
)
from src.datasets import get_dataset, ProMDataset, BrainDataset


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def run_dataset_optim(
    cfg: Dict, ds: Optional[ProMDataset] = None, loss_func: Optional[nn.Module] = None
) -> None:
    if cfg['ensemble_members'] is not None:
        run_dataset_optim_ensemble(cfg, loss_func, members=cfg['ensemble_members'])
        return

    # Initialize objects
    # ----------------------------------------------------------------------------------
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    if loss_func is not None:
        loss_func = loss_func
    else:
        loss_func = nn.MSELoss()

    if ds is None:
        ds = get_dataset(cfg['dataset'], cfg['dataset_root'])

    data_loader = DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
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
        height=ds.img_size,
        width=ds.img_size,
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
        if cfg['use_seg']:
            seg = samples['seg'].to(cfg['device'])
            seg = torch.repeat_interleave(seg, repeats=cfg['bern_samples'], dim=0)
            loss = loss_func(img_mag, seg)
        else:
            loss = loss_func(img_mag, img_batch)

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

        if step == 1 or step % cfg['log_interval'] == 0 or step == cfg['steps']:
            logger.append(
                probs=mask_handler.get_mask_distribution(),
                step=step,
                dense_rate=dense_scheduler.get_dense_rate(),
            )

        if cfg['log_imgs'] > 0 and step % (cfg['steps'] // cfg['log_imgs']) == 0:
            if isinstance(ds, BrainDataset):
                img_rec = torch.abs(img_pred[0, torch.randint(0, 3, (1,))[0]])
                log_masks.append(mask[0, 0].detach().cpu())
            else:
                img_rec = torch.abs(img_pred[0])
                log_masks.append(mask[0].detach().cpu())

            img_low_q = torch.quantile(img, 0.01)
            img_high_q = torch.quantile(img, 0.99)
            img_rec = min_max_normalize(img_rec, img_low_q, img_high_q)
            log_recs.append(img_rec.detach().cpu())

    # ------------------------------------------------------------------------------
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger.write()

    # Construct training progress grid
    if isinstance(ds, BrainDataset):
        img_org = min_max_normalize(img[0, 2], img_low_q, img_high_q)
        log_recs.append(img_org.cpu())
        log_masks.append(torch.zeros_like(img[0, 2]).cpu())

        img_list = log_recs + log_masks
        for i in range(len(img_list)):
            img_list[i] = img_list[i].unsqueeze(0)
    else:
        img_org = min_max_normalize(img[0], img_low_q, img_high_q)
        log_recs.append(img_org.cpu())
        log_masks.append(torch.zeros_like(img[0]).cpu())
        img_list = log_recs + log_masks

    save_image(
        img_list,
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


def run_dataset_optim_ensemble(
    cfg: Dict,
    ds: Optional[ProMDataset] = None,
    loss_func: Optional[nn.Module] = None,
    members: int = 10,
) -> None:
    print('RUNNING ENSEMBLE WITH {} MEMBERS'.format(members))
    # Initialize objects
    # ----------------------------------------------------------------------------------
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    if loss_func is not None:
        loss_func = loss_func
    else:
        loss_func = nn.MSELoss()

    if ds is None:
        ds = get_dataset(cfg['dataset'], cfg['dataset_root'])

    data_loader = DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    data_iter = iter(cycle(data_loader))

    dense_scheduler = DenseRateScheduler(
        target=cfg['dense_target'],
        start_epoch=int(cfg['dense_start'] * cfg['steps']),
        stop_epoch=int(cfg['dense_end'] * cfg['steps']),
    )

    # Create mask handler that also contains scores
    mask_handler = get_mask_handler(
        name=cfg['mask_style'],
        height=ds.img_size,
        width=ds.img_size,
        device=cfg['device'],
    )

    # Initialize optimizer with mask scores
    optimizer = Adam([mask_handler.get_scores()], lr=cfg['learning_rate'])

    # Create constrainer for projection of scores to dense_rate
    constrainer = ScoreConstrainer(mask_handler.get_scores())

    final_scores = []
    for member_idx in range(members):
        print('TRAINING MEMBER IDX {}'.format(member_idx))
        for step in (pbar := tqdm(range(1, cfg['steps'] + 1))):

            samples = next(data_iter)
            img = samples['img']
            img_k = samples['k_space']

            img = img.to(cfg['device'])
            img_k = img_k.to(cfg['device'])

            img_k_batch = torch.repeat_interleave(
                img_k, repeats=cfg['bern_samples'], dim=0
            )
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
            if cfg['use_seg']:
                seg = samples['seg'].to(cfg['device'])
                seg = torch.repeat_interleave(seg, repeats=cfg['bern_samples'], dim=0)
                loss = loss_func(img_mag, seg)
            else:
                loss = loss_func(img_mag, img_batch)

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
        final_scores.append(mask_handler.get_mask_distribution().detach().cpu())

    # ------------------------------------------------------------------------------
    scores = torch.sum(torch.cat(final_scores), dim=(0, 1))
    scores /= members

    os.makedirs(cfg['log_dir'], exist_ok=True)
    torch.save({'scores': [scores.cpu()]}, os.path.join(cfg['log_dir'], 'results.pt'))
