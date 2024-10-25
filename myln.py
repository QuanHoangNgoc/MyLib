from lightning import L
from mytrain import TrainMd, History
from mymodel import ModelMd
from myeval import EvalMd
import torch


class Lbase(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.Loptim, self.Lmodel, self.Lhis = None, None, History()

    def configure_optimizers(self):
        optim = self.Loptim
        return optim

    def training_step(self, batch, batch_idx):
        enc = batch
        for k, v in enc.items():
            if v.dtype in [torch.float32, torch.float16]:
                enc[k] = v.type(torch.float16).requires_grad_(True)
        loss = self.Lmodel(**enc).loss
        self.Lhis.add_history(loss, batch_idx + 1, "train")
        if batch_idx % 10 == 0:
            self.Lhis.print_history("train")
        return loss

    def validation_step(self, batch, batch_idx):
        enc = batch
        loss = self.Lmodel(**enc).loss
        self.Lhis.add_history(loss, batch_idx + 1, "val")
        if batch_idx % 10 == 0:
            self.Lhis.print_history("val")
