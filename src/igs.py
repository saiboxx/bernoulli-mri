from typing import Optional, List

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils import ifft2c


class IGS:
    """Implements Iterative Gradients Sampling from Razumov et al"""

    def __init__(
        self,
        loss_func: Optional[nn.Module] = None,
        device: str = 'cuda',
        num_workers: int = 8,
        batch_size: int = 32,
        use_seg: bool = False,
    ) -> None:
        if loss_func is None:
            self.loss_func = nn.L1Loss()
        else:
            self.loss_func = loss_func

        self.device = torch.device(device)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_seg = use_seg

    @staticmethod
    def get_n(acc_fac: int, img_size: int) -> int:
        return img_size // acc_fac

    def run(self, ds: Dataset, acc_fac: int) -> List[Tensor]:
        dl = DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        img_size = ds[0]['img'].shape[-1]
        n = IGS.get_n(acc_fac, img_size)
        w = torch.zeros(
            img_size,
            dtype=torch.float,
            device=self.device,
        )
        w[img_size // 2] = 1

        w_list = []

        pbar = tqdm(range(n))
        for _ in pbar:

            w.grad = None
            w.requires_grad = True
            for batch in dl:
                img_k = batch['k_space'].to(self.device)

                img_pred = ifft2c(img_k * w + 0.0)
                img_mag = torch.abs(img_pred)

                if self.use_seg:
                    seg = batch['seg'].to(self.device)
                    loss = self.loss_func(img_mag, seg)
                else:
                    img = batch['img'].to(self.device)
                    loss = self.loss_func(img_mag, img)

                loss.backward()

            for i in torch.topk(w.grad, img_size, largest=False).indices:
                if w[i] == 0:
                    w = w.detach()
                    w[i] = 1.0
                    w_list.append(w.clone())
                    pbar.set_description(
                        'select: %d, loss: %.6f' % (i.item(), loss.item())
                    )
                    break
        return w_list
