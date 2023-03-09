import argparse
import os
from typing import Dict, Optional

import torch

from src.run import run_dataset_optim

NUM_RUNS = 10
NUM_MEMBERS = 10

BASE_CONFIG = {
    'dataset': 'knee',
    'dataset_root': '/data',
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
    'log_dir': 'logs/knee',
    'log_imgs': 10,
    'log_interval': 100,
    'seed': None,
}

def main(cfg: Dict, mask_style: str = 'f', acc_fac: Optional[int] = None) -> None:
    cfg['mask_style'] = mask_style

    print('MASK STYLE: {}'.format(cfg['mask_style']))
    if mask_style == 'f':
        prefix = '2d'
    else:
        prefix = '1d'

    if acc_fac is None:
        acc_facs = [4, 8, 16, 32]
    else:
        acc_facs = [acc_fac]

    for run_idx in range(1, 1 + NUM_RUNS):
        for acc_fac in acc_facs:
            for i in range(1, 1 + NUM_MEMBERS):
                print('RUN {} | ACC FAC {} | MEMBER {}'.format(run_idx, acc_fac, i))
                cfg['dense_target'] = 1 / acc_fac
                cfg['log_dir'] = os.path.join(
                    'logs',
                    'knee_' + prefix + '_a' + str(acc_fac),
                    'knee_r' + str(run_idx) + '_m' + str(i)
                )

                if os.path.exists(cfg['log_dir']):
                    continue

                run_dataset_optim(cfg)

    for run_idx in range(1, 1 + NUM_RUNS):
        for acc_fac in acc_facs:
            path_stem = f'logs/knee_' + prefix + f'_a{acc_fac}/knee_r{run_idx}_m'

            paths = [path_stem + str(i) + '/results.pt' for i in range(1, NUM_MEMBERS + 1)]
            scores = [torch.load(f)['scores'][-1].cuda() for f in paths]
            scores /= NUM_MEMBERS

            scores_sum = torch.sum(torch.cat(scores), dim=(0, 1))

            new_path = f'logs/knee_' + prefix + f'_a{acc_fac}/knee_r{run_idx}.pt'
            torch.save({
                'scores': [scores_sum.cpu()]
            }, new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', '-m', type=str)
    parser.add_argument('--acc', '-a', type=int)
    args = parser.parse_args()
    main(BASE_CONFIG, args.mask, args.acc)