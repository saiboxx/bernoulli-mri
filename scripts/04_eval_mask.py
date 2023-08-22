import torch
from torch.utils.data import DataLoader

from src.datasets import get_dataset
from src.distribution import SoftBernoulliSampler
from src.utils import get_top_k_mask, get_reconstruction_metrics, \
    get_segmentation_metrics, get_unet

RESULTS_FILE = 'PATH/RESULTS.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POSTERIOR_MODE = True
ACC_FAC = 8
DATASET_NAME = 'knee'
DATASET_ROOT = 'data'
SEGMENTATION_METRICS = False

def main() -> None:
    # Get dataset and loader
    ds = get_dataset(DATASET_NAME, train=False)
    dl = DataLoader(ds, batch_size=64, num_workers=16)

    # Load target fitted distribution
    score = torch.load(RESULTS_FILE)['scores'][-1].squeeze().to(DEVICE)

    # Sample mask from posterior mode or random
    if POSTERIOR_MODE:
        mask = get_top_k_mask(score, ACC_FAC)
    else:
        mask = SoftBernoulliSampler().sample(score)
        mask = torch.nan_to_num(mask)

    # Compute metrics for reconstruction or segmentation
    if SEGMENTATION_METRICS:
        model = get_unet('acdc')
        dice, iou = get_segmentation_metrics(dl, model, DEVICE, mask)
        print('DICE: {:.3f} | IOU {:.3f} '.format(dice, iou))

    else:
        psnr, ssim, nmse = get_reconstruction_metrics(dl, mask, DEVICE)
        print('PSNR: {:.3f} | SSIM {:.3f} | NMSE {:.3f}'.format(psnr, ssim, nmse))


if __name__ == '__main__':
    main()