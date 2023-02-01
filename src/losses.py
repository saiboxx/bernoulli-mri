import torch
from torch import nn, Tensor
from torch.nn.functional import conv2d
from kornia.losses import SSIMLoss

class SobelLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        kernel_x = torch.tensor(
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1],
            ]
        )

        kernel_y = torch.tensor(
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1],
            ]
        )

        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).float()
        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0).float()

        self.kernel_x: Tensor
        self.kernel_y: Tensor
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

        self.padding = 1
        self.loss_func = nn.MSELoss()

    def compute_edges(self, x: Tensor) -> Tensor:
        x_kx = conv2d(x, self.kernel_x, padding=self.padding)
        x_ky = conv2d(x, self.kernel_y, padding=self.padding)
        return torch.sqrt(torch.pow(x_kx, 2) + torch.pow(x_ky, 2) + 1e-10)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        pred_edges = self.compute_edges(prediction)
        target_edges = self.compute_edges(target)

        return self.loss_func(pred_edges, target_edges)


class SobelMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

        self.mse = nn.MSELoss()
        self.sobel = SobelLoss()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        mse = self.mse(prediction, target)
        sob = self.sobel(prediction, target)
        return self.alpha * mse + (1 - self.alpha) * sob


class SSIMMSELoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ssim_window: int = 1) -> None:
        super().__init__()
        self.alpha = alpha

        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(ssim_window)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        mse = self.mse(prediction, target)
        ssim = self.ssim(prediction, target)
        return self.alpha * mse + (1 - self.alpha) * ssim