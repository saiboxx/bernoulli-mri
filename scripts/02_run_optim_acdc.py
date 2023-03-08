import argparse
import os
from typing import Dict, Optional

import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from src.run import run_dataset_optim
from src.losses import SegmentationProxyLoss

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

NUM_RUNS = 10
NUM_MEMBERS = 10

BASE_CONFIG = {
    'dataset': 'acdc',
    'dataset_root': '/data/core-rad/data',
    'batch_size': 32,
    'steps': 2500,
    'use_seg': True,
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
}

def main(cfg: Dict, mask_style: str, acc_fac: Optional[int] = None) -> None:
    cfg['mask_style'] = mask_style

    print('MASK STYLE: {}'.format(cfg['mask_style']))
    if mask_style == 'f':
        prefix = '2d'
    else:
        prefix = '1d'

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).cuda()

    sd = torch.load('models/acdc_unet.pt')
    model.load_state_dict(sd['model'])

    loss_func = DiceCELoss(
        softmax=True,
        include_background=False,
        to_onehot_y=True
    )

    loss_func = SegmentationProxyLoss(model=model, seg_loss_func=loss_func).cuda()

    if acc_fac is None:
        acc_facs = [8, 16, 32]
    else:
        acc_facs = [acc_fac]

    for run_idx in range(1, 1 + NUM_RUNS):
        for acc_fac in acc_facs:
            for i in range(1, 1 + NUM_MEMBERS):
                print('RUN {} | ACC FAC {} | MEMBER {}'.format(run_idx, acc_fac, i))
                cfg['dense_target'] = 1 / acc_fac
                cfg['log_dir'] = os.path.join(
                    'logs',
                    'acdc_' + prefix + '_a' + str(acc_fac),
                    'acdc_r' + str(run_idx) + '_m' + str(i)
                )

                if os.path.exists(cfg['log_dir']):
                    continue

                run_dataset_optim(cfg, loss_func=loss_func)

    for run_idx in range(1, 1 + NUM_RUNS):
        for acc_fac in acc_facs:
            path_stem = f'logs/acdc_' + prefix + f'_a{acc_fac}/acdc_r{run_idx}_m'

            paths = [path_stem + str(i) + '/results.pt' for i in range(1, NUM_MEMBERS + 1)]
            scores = [torch.load(f)['scores'][-1].cuda() for f in paths]

            scores_sum = torch.sum(torch.cat(scores), dim=(0, 1))

            new_path = f'logs/acdc_' + prefix + f'_a{acc_fac}/acdc_r{run_idx}.pt'
            torch.save({
                'scores': [scores_sum.cpu()]
            }, new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', '-m', type=str)
    parser.add_argument('--acc', '-a', type=int)
    args = parser.parse_args()
    main(BASE_CONFIG, args.mask, args.acc)