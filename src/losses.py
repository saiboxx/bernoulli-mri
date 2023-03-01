from torch import nn, Tensor


class SegmentationProxyLoss(nn.Module):
    def __init__(self, model: nn.Module, seg_loss_func: nn.Module) -> None:
        super().__init__()

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.loss_func = seg_loss_func

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        seg_pred = self.model(prediction)
        return self.loss_func(seg_pred, target)