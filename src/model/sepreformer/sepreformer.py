import torch
import torch.nn.functional as F
from torch import nn

from .modules import AudioDecoder, AudioEncoder, InputLayer, OutputLayer
from .separator import Separator


class SepReformer(nn.Module):
    """
    Separate and Reconstruct: Asymmetric Encoder-Decoder for Speech Separation
    https://arxiv.org/pdf/2406.05983
    """

    def __init__(
        self,
        num_spks=2,  # S
        num_stages=4,  # R
        audio_enc_dim=256,  # Fo = N
        audio_kernel_size=16,  # L
        audio_stride=4,  # H
        feature_dim=64,  # F
        num_heads=8,
        dropout_p=0.1,
        kernel_size=65,
        dconv_kernel_size=5,
        pe_maxlen=2000,
    ):
        super().__init__()
        self.num_spks = num_spks

        self.audio_encoder = AudioEncoder(1, audio_enc_dim, audio_kernel_size, stride=audio_stride, bias=False)
        self.input_layer = InputLayer(audio_enc_dim, feature_dim, 1, bias=False)

        self.separator = Separator(
            num_spks,
            num_stages,
            num_heads,
            in_channels=feature_dim,
            dropout_p=dropout_p,
            kernel_size=kernel_size,
            dconv_kernel_size=dconv_kernel_size,
            pe_maxlen=pe_maxlen,
        )

        self.output_layer = OutputLayer(feature_dim, audio_enc_dim, num_spks=num_spks)
        self.audio_decoder = AudioDecoder(audio_enc_dim, 1, audio_kernel_size, stride=audio_stride, bias=False)

        # Intermediate outputs
        self.output_aux = nn.ModuleList()
        self.decoder_aux = nn.ModuleList()
        for _ in range(num_stages):
            self.output_aux.append(OutputLayer(feature_dim, audio_enc_dim, num_spks=num_spks, masking=True))
            self.decoder_aux.append(AudioDecoder(audio_enc_dim, 1, audio_kernel_size, stride=audio_stride, bias=False))

    def forward(self, mixture, **batch):
        """
        Args:
            mixture (Tensor): mixed audio (B, T).
        Returns:
            preds (Tensor): predicted separated sources (B, C, T).
            preds_aux (List[Tensor]): predicted separated sources from intermediate layers (B, C, T).
        """
        spec = self.audio_encoder(mixture)
        feature = self.input_layer(spec)

        # feature_sources [B*S, F, T], intermediate_outputs [[B*S, F, T/2] [B*S, F, T/4]...]
        feature_sources, intermediate_outputs = self.separator(feature)
        spec_separated = self.output_layer(feature_sources, spec)  # [S, B, N, T]

        preds = torch.stack([self.audio_decoder(spec_separated[i]) for i in range(self.num_spks)], dim=1)

        # For auxiliary loss
        preds_aux_list = []
        specT = spec.size(-1)
        T = mixture.size(-1)
        for i, (output_layer, decoder_layer) in enumerate(zip(self.output_aux, self.decoder_aux)):
            feature_sources = F.upsample(intermediate_outputs[i], specT)
            spec_separated = output_layer(feature_sources, spec)

            preds_aux = torch.stack([decoder_layer(spec_separated[j])[..., :T] for j in range(self.num_spks)], dim=1)
            preds_aux_list.append(preds_aux)

        return {"preds": preds, "preds_aux": preds_aux_list}
