import torch
from torch import Tensor


class SoftBernoulliSampler:
    """Soft Bernoulli distribution that allows backpropagation."""

    def __init__(self, eps: float = 1e-20):
        """Initialize a SoftBernoulliSampler object."""
        self.eps = eps

    def sample(
        self, scores: Tensor, temperature: float = 0.1, hard: bool = True
    ) -> Tensor:
        """
        Sample from the distribution.
        The output will be the shape of `scores`. Precisely, for every entry in a
        `scores` tensor one sample is drawn independently from the distribution.
        The value of `scores` should be [0, 1] and is the probability of the random
        variable being 1. The `temperature` parameter describes the softness of this
        backpropagatable approximation. A low `temperature` will produce events close
        to {0, 1}, whereas a high `temperature` has more degrees of freedom.
        """
        uniform_0 = torch.rand_like(scores)
        uniform_1 = torch.rand_like(scores)
        gumbel_subtraction = -torch.log(
            torch.log(uniform_0 + self.eps) / torch.log(uniform_1 + self.eps) + self.eps
        )
        score_logits = torch.log(scores + self.eps) - torch.log(1.0 - scores + self.eps)
        y_soft = torch.sigmoid((score_logits + gumbel_subtraction) / temperature)

        if hard:
            y_hard = torch.where(y_soft > 0.5, 1.0, 0.0)
            return y_hard - y_soft.detach() + y_soft
        else:
            return y_soft


#%%
