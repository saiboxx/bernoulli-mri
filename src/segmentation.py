import os

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import ACDCDataset, BrainDataset, KneeDataset


def train_segmentation_net(
        model: nn.Module,
        optimizer: Optimizer,
        loss_func: nn.Module,
        dataset: str,
        save_dir: str,
        save_name: str,
        batch_size: int,
        epochs: int,
        num_workers: int,
        device: torch.device,
        dataset_root: str = 'data',
        seed: int = 42,
) -> None:

    # SETUP
    # ----------------------------------------------------------------------------------
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    if dataset == 'acdc':
        train_ds = ACDCDataset(os.path.join(dataset_root, 'ACDC'), train=True)
        val_ds = ACDCDataset(os.path.join(dataset_root, 'ACDC'), train=False)
    elif dataset == 'brain':
        train_ds = BrainDataset(os.path.join(dataset_root, 'Task01_BrainTumour'), train=True)
        val_ds = BrainDataset(os.path.join(dataset_root, 'Task01_BrainTumour'), train=False)
    elif dataset == 'knee':
        train_ds = KneeDataset(os.path.join(dataset_root, 'knee_fastmri'), train=True)
        val_ds = KneeDataset(os.path.join(dataset_root, 'knee_fastmri'), train=False)
    else:
        raise ValueError('Dataset {} unknown.'.format(dataset))


    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    # TRAINING LOOP
    # ----------------------------------------------------------------------------------
    best_loss = 9999999999.
    patience = 5
    patience_count = 0

    for ep in tqdm(range(1, epochs + 1)):
        for batch in tqdm(train_loader, leave=False):
            img = batch['img'].to(device)
            seg = batch['seg'].to(device)

            optimizer.zero_grad()
            pred = model(img)
            loss = loss_func(pred, seg)
            loss.backward()
            optimizer.step()

    # VALIDATION LOOP
    # ----------------------------------------------------------------------------------
        with torch.no_grad():
            val_loss = 0.
            for batch in tqdm(val_loader, leave=False):
                img = batch['img'].to(device)
                seg = batch['seg'].to(device)

                pred = model(img)
                loss = loss_func(pred, seg)

                val_loss += float(loss)

            if val_loss <= best_loss:
                patience_count = 0
                best_loss = val_loss
                tqdm.write(
                    'New best loss: {:.3f}\t|\tCheckpoint saved in epoch {:4d}.'
                    .format(best_loss, ep))
                path = os.path.join(save_dir, save_name)
                torch.save({
                    'model': model.state_dict(),
                    'ep': ep,
                    'batch_size': batch_size,
                    'val_loss': val_loss,
                }, path)

            else:
                patience_count += 1

            if patience_count >= patience:
                tqdm.write(
                'Model has not improved in the last {} epochs. ' \
                'Training is stopped after {} epochs'.format(patience_count, ep)
                )
                return

