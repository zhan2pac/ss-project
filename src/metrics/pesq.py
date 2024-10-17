import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(self, fs: int, mode: str, device: str, *args, **kwargs):
        """
        Use TorchMetrics PerceptualEvaluationSpeechQuality function inside.

        Args:
            fs (int): sampling frequency, should be 16000 or 8000 (Hz).
            mode (str): 'wb' (wide-band) or 'nb' (narrow-band).
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        metric = PerceptualEvaluationSpeechQuality(fs, mode, *args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(
        self, preds: torch.Tensor, labels: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Calculate PESQ metric.

        Args:
            preds (Tensor): model predictions (..., Time).
            labels (Tensor): ground-truth audio (..., Time).
        Returns:
            metrics (Tensor): calculated PESQ.
        """

        return self.metric(preds, labels)
