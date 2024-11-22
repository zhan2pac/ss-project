from torch.optim import lr_scheduler


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    """
    ReduceLROnPlateau scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(self, SI_SNRi, **batch):
        super().step(SI_SNRi)
