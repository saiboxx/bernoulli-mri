from abc import ABC, abstractmethod
import os
from typing import Tuple

import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
from tqdm import tqdm
import torch
from torch import Tensor, nn
from torchvision.transforms.functional import center_crop
from torchvision.utils import save_image

from src.config import Configuration
from src.constraint import ScoreConstrainer
from src.dense_rate import DenseRateScheduler
from src.distribution import SoftBernoulliSampler
from src.utils import load_h5_slice, get_temperature, ifft2c, fft2c, plot_heatmap


class MaskOptimizer:
    def __init__(self, cfg: Configuration) -> None:
        self.cfg = cfg
        self.dense_scheduler = DenseRateScheduler(
            target=cfg.dense_target,
            start_epoch=int(cfg.dense_start * cfg.steps),
            stop_epoch=int(cfg.dense_end * cfg.steps),
        )

    def run(self, loss_func: nn.Module) -> Tensor:
        # Load data from file
        img, img_k = self.load_data()
        img = img.unsqueeze(0)
        img_k = img_k.unsqueeze(0)

        # Create mask handler that also contains scores
        mask_handler = get_mask_handler(
            name=self.cfg.mask_style,
            height=img.shape[-2],
            width=img.shape[-1],
            device=self.cfg.device,
        )

        # Initialize optimizer with mask scores
        optimizer = Adam([mask_handler.get_scores()], lr=self.cfg.learning_rate)

        # Create constrainer for projection of scores to dense_rate
        constrainer = ScoreConstrainer(mask_handler.get_scores())

        log_recs = []
        log_masks = []
        img_min = torch.min(img)
        img_max = torch.max(img)

        for step in (pbar := tqdm(range(1, self.cfg.steps + 1))):

            # Get temperature for categorical "softness"
            temperature = get_temperature(step, self.cfg.steps)

            # Map scores to valid probability space
            dense_rate = self.dense_scheduler.get_dense_rate()
            constrainer.constrain(dense_rate=dense_rate)
            self.dense_scheduler.advance()

            # Sample from distribution
            mask = mask_handler.sample_mask(
                temperature=temperature, num_samples=self.cfg.bern_samples
            )

            # Compute image with mask
            img_pred = ifft2c(img_k * mask + 0.0)
            img_mag = torch.abs(img_pred)

            # Compute loss between full and undersampled image
            img_batch = img.expand(self.cfg.bern_samples, -1, -1, -1)
            loss = loss_func(img_mag, img_batch)

            # Optimize scores
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                pbar.set_description(
                    'L: {:.5f} | D: {:.3f}'.format(
                        float(loss), float(torch.mean(mask_handler.get_scores()))
                    )
                )

            num_imgs = self.cfg.log_imgs
            if num_imgs > 0 and step % (self.cfg.steps // num_imgs) == 0:
                img_rec = torch.abs(img_pred[0])
                img_rec = (img_rec - img_min) / (img_max - img_min)
                img_rec.clamp_(0, 1)

                log_recs.append(img_rec.detach().cpu())
                log_masks.append(mask[0].detach().cpu())

        # ------------------------------------------------------------------------------

        # Construct training progress grid
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        img_org = (img - img_min) / (img_max - img_min)
        log_recs.append(img_org.squeeze(0).cpu())
        log_masks.append(torch.zeros_like(img).squeeze(0).cpu())

        save_image(
            log_recs + log_masks,
            fp=self.cfg.log_dir + '/progress.png',
            nrow=len(log_recs),
        )

        # Construct heatmap
        plot_heatmap(
            mask_handler.get_mask_distribution(),
            save_path=self.cfg.log_dir + '/heatmap.png',
        )

        # Construct histogram of scores
        plt.hist(mask_handler.get_scores().detach().cpu().flatten(), bins=20)
        plt.savefig(self.cfg.log_dir + '/histogram.png')

    def load_data(self) -> Tuple[Tensor, Tensor]:
        img_k = load_h5_slice(self.cfg.file_path, self.cfg.slice_idx, self.cfg.coil_idx)
        img = ifft2c(img_k)

        if self.cfg.cropping is not None:
            img = center_crop(img, list(self.cfg.cropping))
            img_k = fft2c(img)

        return torch.abs(img).to(self.cfg.device), img_k.to(self.cfg.device)


class BaseMask(ABC):
    def __init__(
        self, height: int, width: int, device: torch.device, channels: int = 1
    ) -> None:
        self.sampler = SoftBernoulliSampler()

        self.height = height
        self.width = width
        self.channels = channels
        self.device = device

        self.scores = self.initialize_scores()
        self.scores.requires_grad = True

    def get_scores(self) -> Tensor:
        return self.scores

    @abstractmethod
    def initialize_scores(self) -> Tensor:
        pass

    @abstractmethod
    def get_mask_distribution(self) -> Tensor:
        pass

    def _draw_samples(self, temperature: float = 0, num_samples: int = 1) -> Tensor:
        batch_scores = self.scores.expand(num_samples, -1, -1, -1)
        return self.sampler.sample(batch_scores, temperature)

    @abstractmethod
    def sample_mask(self, temperature: float = 0, num_samples: int = 1) -> Tensor:
        pass


class FullMask(BaseMask):
    def initialize_scores(self) -> Tensor:
        shape = (1, self.channels, self.height, self.width)
        scores = torch.rand(shape, dtype=torch.float, device=self.device)
        return scores

    def get_mask_distribution(self) -> Tensor:
        return self.get_scores()

    def sample_mask(self, temperature: float = 0, num_samples: int = 1) -> Tensor:
        return self._draw_samples(temperature, num_samples)


class HorizontalMask(BaseMask):
    def initialize_scores(self) -> Tensor:
        shape = (1, self.channels, 1, self.width)
        scores = torch.rand(shape, dtype=torch.float, device=self.device)
        return scores

    def get_mask_distribution(self) -> Tensor:
        scores = self.get_scores()
        return scores.expand(-1, -1, self.height, -1)

    def sample_mask(self, temperature: float = 0, num_samples: int = 1) -> Tensor:
        horizontal_samples = self._draw_samples(temperature, num_samples)
        return horizontal_samples.expand(-1, -1, self.height, -1)


class VerticalMask(BaseMask):
    def initialize_scores(self) -> Tensor:
        shape = (1, self.channels, self.height, 1)
        scores = torch.rand(shape, dtype=torch.float, device=self.device)
        return scores

    def get_mask_distribution(self) -> Tensor:
        scores = self.get_scores()
        return scores.expand(-1, -1, -1, self.width)

    def sample_mask(self, temperature: float = 0, num_samples: int = 1) -> Tensor:
        vertical_samples = self._draw_samples(temperature, num_samples)
        return vertical_samples.expand(-1, -1, -1, self.width)


def get_mask_handler(
    name: str, height: int, width: int, device: torch.device, channels: int = 1
) -> BaseMask:
    if name == 'f':
        cls = FullMask
    elif name == 'h':
        cls = HorizontalMask
    elif name == 'v':
        cls = VerticalMask
    else:
        raise ValueError('Mask name "{}" is unknown.'.format(name))

    return cls(height, width, device, channels)
