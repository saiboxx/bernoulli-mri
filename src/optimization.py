from abc import ABC, abstractmethod

import torch
from torch import Tensor

from src.distribution import SoftBernoulliSampler


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
