from torch import nn

from .normalization import CumulativeLN, GlobalLN
from .temporalconvnet import TemporalConvNet


class Separator(nn.Module):
    def __init__(
        self,
        c_sources,
        dim,
        num_repeats,
        num_blocks,
        bottle_dim,
        hidden_dim,
        kernel_size,
        norm,
        causal,
        masks_act,
    ):
        super().__init__()
        self.c_sources = c_sources
        self.dim = dim

        masks_act_module = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}

        self.layer_norm = CumulativeLN(dim)
        self.bottleneck_conv1x1 = nn.Conv1d(dim, bottle_dim, kernel_size=1, bias=False)
        self.network = TemporalConvNet(
            num_repeats,
            num_blocks,
            in_channels=bottle_dim,
            hid_channels=hidden_dim,
            kernel_size=kernel_size,
            norm=norm,
            causal=causal,
        )

        self.prelu = nn.PReLU()
        self.masks_conv1x1 = nn.Conv1d(bottle_dim, c_sources * dim, kernel_size=1, bias=False)
        self.masks_act = masks_act_module[masks_act]

    def forward(self, x):
        """
        Args:
            x (Tensor): input of shape (batch_size, dim, time)
        Returns:
            masks (Tensor): masks of shape (batch_size, c_sources, dim, time)
        """
        batch_size = x.size(0)
        out = self.layer_norm(x)
        out = self.bottleneck_conv1x1(out)

        out = self.network(out)  # [batch_size, bottle_dim, time]

        out = self.prelu(out)

        masks = self.masks_conv1x1(out)  # [batch_size, c_sources * dim, time]
        masks = masks.view(batch_size, self.c_sources, self.dim, -1)
        masks = self.masks_act(masks)

        return masks


class ConvTasNet(nn.Module):
    """
    Convolutional Time-domain Audio Separation Network.
    Applies directly on waveforms. Encoder and Decoder are learned instead of STFT and iSTFT.
    Separation masks are extracted through Temporal Convolutional Network.

    https://arxiv.org/pdf/1809.07454
    """

    def __init__(
        self,
        c_sources=2,
        dim=512,
        length_seg=16,
        bottle_dim=128,
        hidden_dim=512,
        kernel_size=3,
        num_blocks=8,
        num_repeats=3,
        norm="global_ln",
        causal=False,
        masks_act="sigmoid",
    ):
        """
        Args:
            c_sources (int): number of target sources (C).
            dim (int): number of generated features from segment (N).
            length_seg (int): length of the segment from input audio (L).
            bottle_dim (int): bottleneck dimension (B).
            hidden_dim (int): hidden dimension (H).
            kernel_size (int): size of convolution kernel (P).
            num_blocks (int): number of dilated blocks (X).
            num_repeats (int): number of repeated temporal layers (R).
            norm (str): normalization type (either "global_ln", "cumulative_ln", "batchnorm").
            causal (int): causality configuration (only CausalLN can be used if causal=True).
            masks_act (str): mask activation type (either "relu", "softmax").
        """
        super().__init__()
        self.c_sources = c_sources
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, dim, kernel_size=length_seg, stride=length_seg // 2),
            nn.ReLU(),  # non-negative representation
        )

        self.separator = Separator(
            c_sources,
            dim,
            num_repeats,
            num_blocks,
            bottle_dim,
            hidden_dim,
            kernel_size,
            norm,
            causal,
            masks_act,
        )

        self.decoder = nn.ConvTranspose1d(dim, 1, kernel_size=length_seg, stride=length_seg // 2)

    def forward(self, mixture, **batch):
        """
        Args:
            mixture (Tensor): mixed audio (batch_size, time).
        Returns:
            output (dict): output dict containing predicted separated sources (batch_size, c_sources, time).
        """
        batch_size = mixture.size(0)
        mixture = mixture.unsqueeze(1)
        mixture_emb = self.encoder(mixture)  # [batch_size, dim, time]

        masks = self.separator(mixture_emb)  # [batch_size, c_sources, dim, time]

        sources_emb = mixture_emb.unsqueeze(1) * masks
        sources_emb = sources_emb.view(batch_size * self.c_sources, self.dim, -1)

        sources = self.decoder(sources_emb)  # [batch_size*c_sources, 1, time]
        sources = sources.view(batch_size, self.c_sources, -1)

        return {"preds": sources}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
