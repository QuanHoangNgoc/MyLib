import os
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from utModule import ut


class TrainModule:
    def __init__(self) -> None:
        pass

    pattern_loader = {
        "num_workers": os.cpu_count(),
        "prefetch_factor": 2,
        "pin_memory": True,
    }

    def get_dataloader(torch_ds, bs, shuffle, pdic=pattern_loader):
        torch_ds.set_format("torch")
        return DataLoader(torch_ds, batch_size=bs, shuffle=shuffle, **pdic)

    def train_on_model(model, device=None, gradient_checkpointing=False):
        if device is not None:
            model.to(device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model.train()

    def get_adamw_from_model(model, base_lr=5e-5):
        param_groups = []
        lr = base_lr
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_groups.append(
                    {
                        "params": param,
                        "lr": lr,
                    }
                )
        optimizer = AdamW(params=param_groups)
        return optimizer


class History:
    def __init__(self) -> None:
        self.track = {
            "train": {
                "total": 0.0,
                "loss": [],
            },
            "val": {
                "total": 0.0,
                "loss": [],
            },
        }

    def add_history(self, loss, step, key):
        if step == 1:
            self.track[key]["total"] = 0.0
            self.track[key]["loss"].append([])
        self.track[key]["total"] += loss.item()
        self.track[key]["loss"][-1].append(loss.item())

    def print_history(self, key):
        step = len(self.track[key]["loss"][-1])
        ut.mess(f"[step {step}], {key} loss: {self.track[key]['total'] / step:.4f}")
