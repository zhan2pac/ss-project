from collections import defaultdict
from pathlib import Path
from typing import Optional

import gdown
import torch
from torch import Tensor, nn

from src.model.rtfs_net.video_model import ResVideoModel
from src.utils.init_utils import ROOT_PATH


class VideoEncoder(nn.Module):
    """
    Video encoder block using pre-trained model
    """

    URL = "https://drive.google.com/uc?id=1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD"
    DIR = ROOT_PATH / "data/models"
    PATH = DIR / "video_model.pth"

    def __init__(self):
        """
        Args:
        """
        super(VideoEncoder, self).__init__()

        self.encoder = ResVideoModel()
        self._load_pretrained()

    def forward(self, video: Tensor) -> Tensor:
        """
        Args:
            video (Tensor): video to encode (B, C, L, H, W).
        Return:
            encoded (Tensor): encoded video (B, C, L).
        """
        encoded = self.encoder(video)  # (B, C, L)

        return encoded

    def _load_pretrained(self):
        """
        Load pre-trained model
        """
        self._download_model()
        pretrained_weights = torch.load(str(self.PATH))["model_state_dict"]

        model_dict = self.encoder.state_dict()
        updated_dict = defaultdict()

        for k, v in pretrained_weights.items():
            if "tcn" in k:
                continue

            updated_dict[k] = v

        model_dict.update(updated_dict)
        self.encoder.load_state_dict(model_dict)

        for p in self.encoder.parameters():
            p.requires_grad = False

    def _download_model(self):
        path = self.PATH
        if path.exists():
            return
        self.DIR.mkdir(parents=True, exist_ok=True)

        gdown.download(self.URL, str(path))
