import argparse
import os
import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from src.losses import SegmentationProxyLoss
from src.igs import IGS
from src.datasets import ACDCDataset, BrainDataset, KneeDataset

NUM_WORKERS = 16
DATASET_ROOT = '/data'
LOG_DIR = 'logs/IGS'

def get_unet(dataset_name: str) -> UNet:

    if dataset_name == 'acdc':
        in_channels = 1
        model_path = 'models/acdc_unet.pt'
    elif dataset_name == 'brain':
        in_channels = 4
        model_path = 'models/brain_unet.pt'
    else:
        raise ValueError('Dataset name {} is unknown.'.format(dataset_name))

    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )

    sd = torch.load(model_path)
    model.load_state_dict(sd['model'])
    return model


def main(idx: int = 1) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # RECONSTRUCTION
    # ----------------------------------------------------------------------------------
    ds = KneeDataset(os.path.join(DATASET_ROOT, 'knee_fastmri'), train=True)
    igs = IGS(num_workers=NUM_WORKERS)
    w_list = igs.run(ds, acc_fac=4)
    w_full = torch.stack(w_list).cpu()
    torch.save(w_full, os.path.join(LOG_DIR, 'igs_knee_' + str(idx) + '.pt'))
    # ----------------------------------------------------------------------------------
    # SEGMENTATION
    # ----------------------------------------------------------------------------------
    seg_loss = DiceCELoss(
        softmax=True,
        include_background=False,
        to_onehot_y=True
    )

    model = get_unet('acdc')
    loss_func = SegmentationProxyLoss(model=model, seg_loss_func=seg_loss).to('cuda')
    ds = ACDCDataset(os.path.join(DATASET_ROOT, 'ACDC'), train=True)
    igs = IGS(
        loss_func=loss_func,
        num_workers=NUM_WORKERS,
        use_seg=True
    )
    w_list = igs.run(ds, acc_fac=8)
    w_full = torch.stack(w_list).cpu()
    torch.save(w_full, os.path.join(LOG_DIR, 'igs_acdc_seg_' + str(idx) + '.pt'))

    model = get_unet('brain')
    loss_func = SegmentationProxyLoss(model=model, seg_loss_func=seg_loss).to('cuda')
    ds = BrainDataset(os.path.join(DATASET_ROOT, 'Task01_BrainTumour'), train=True)
    igs = IGS(
        loss_func=loss_func,
        num_workers=NUM_WORKERS,
        use_seg=True
    )
    w_list = igs.run(ds, acc_fac=8)
    w_full = torch.stack(w_list).cpu()
    torch.save(w_full, os.path.join(LOG_DIR, 'igs_brats_seg_' + str(idx) + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', '-i', type=int)
    args = parser.parse_args()
    main(args.idx)
    main()