import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(self, fs: int, device: str, audio_only: bool, *args, **kwargs):
        """
        Use TorchMetrics ShortTimeObjectiveIntelligibility function inside.

        Args:
            fs (int): sampling frequency (Hz).
            device (str): device for the metric calculation (and tensors).
            audio_only (bool): use permute technic to calculate metric.
        """
        super().__init__(*args, **kwargs)

        metric = ShortTimeObjectiveIntelligibility(fs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.audio_only = audio_only

    def __call__(self, preds: torch.Tensor, sources: torch.Tensor, **batch) -> torch.Tensor:
        """
        Calculate PESQ metric.

        Args:
            preds (Tensor): model predictions (..., Time).
            sources (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated STOI.
        """

        if self.audio_only:
            sources_swap = sources[:, [1, 0]]
            metric_value = torch.max(
                self.metric(preds, sources),
                self.metric(preds, sources_swap),
            )
        else:
            metric_value = self.metric(preds, sources)

        return metric_value
