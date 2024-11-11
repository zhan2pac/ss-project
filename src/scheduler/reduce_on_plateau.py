from torch.optim import lr_scheduler


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    """
    ReduceLROnPlateau scheduler
    """

    def step(self, SI_SNRi, **batch):
        super().step(SI_SNRi)
