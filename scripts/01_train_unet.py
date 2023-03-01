# +
import torch
import torch.cuda
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torch.optim import Adam

from src.segmentation import train_segmentation_net

def main():
    device = torch.device('cuda')

    model = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    loss_func = DiceCELoss(
        softmax=True,
        include_background=False,
        to_onehot_y=True
    )

    train_segmentation_net(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        dataset='brain',
        save_dir='models',
        save_name='brain_base.pt',
        batch_size=256,
        epochs=1000,
        num_workers=8,
        device=device,
        dataset_root='/data/core-rad/data',
        seed=42,
    )


if __name__ == '__main__':
    main()
