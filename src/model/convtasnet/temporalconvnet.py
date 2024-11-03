from torch import nn

from .normalization import CumulativeLN, GlobalLN


class TemporalBlock(nn.Module):
    """
    1D Conv Block design as described in ConvTasNet paper.
    """

    def __init__(self, in_channels, hid_channels, kernel_size, dilation, norm="global_ln", causal=False):
        super().__init__()

        norm_module = {"global_ln": GlobalLN, "cumulative_ln": CumulativeLN, "bn": nn.BatchNorm1d}

        self.in_conv1x1 = nn.Conv1d(in_channels, hid_channels, kernel_size=1)
        self.act1 = nn.PReLU()
        self.norm1 = norm_module[norm](hid_channels)

        pad = dilation * (kernel_size - 1)
        if causal:
            self.padder = nn.ConstantPad1d((pad, 0), value=0.0)
        else:
            self.padder = nn.ConstantPad1d((pad // 2, pad // 2), value=0.0)

        self.depthwise_conv = nn.Conv1d(hid_channels, hid_channels, kernel_size, dilation=dilation, groups=hid_channels)
        self.act2 = nn.PReLU()
        self.norm2 = norm_module[norm](hid_channels)

        self.out_conv1x1 = nn.Conv1d(hid_channels, in_channels, kernel_size=1)
        self.skip_conv1x1 = nn.Conv1d(hid_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): input of shape (batch_size, dim, time)
        """
        res = self.in_conv1x1(x)
        res = self.act1(res)  # should be norm before activation?
        res = self.norm1(res)

        res = self.padder(res)

        res = self.depthwise_conv(res)
        res = self.act2(res)
        res = self.norm2(res)

        skip = self.skip_conv1x1(res)
        res = self.out_conv1x1(res)

        return x + res, skip


class TemporalLayer(nn.Module):
    """
    Stacked temporal blocks with increasing dilation factor.
    """

    def __init__(self, num_blocks, in_channels, hid_channels, kernel_size, norm="global_ln", causal=False):
        super().__init__()

        dilated_blocks = [
            TemporalBlock(in_channels, hid_channels, kernel_size, dilation=2**i, norm=norm, causal=causal)
            for i in range(num_blocks)
        ]
        self.dilated_blocks = nn.ModuleList(dilated_blocks)

    def forward(self, x):
        skip_sum = 0.0
        for block in self.dilated_blocks:
            x, skip = block(x)
            skip_sum += skip

        return x, skip_sum


class TemporalConvNet(nn.Module):
    """
    Stacked temporal layers.
    """

    def __init__(self, num_repeats, num_blocks, in_channels, hid_channels, kernel_size, norm="global_ln", causal=False):
        super().__init__()

        layers = [
            TemporalLayer(num_blocks, in_channels, hid_channels, kernel_size, norm=norm, causal=causal)
            for _ in range(num_repeats)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip_sum = 0.0
        for layer in self.layers:
            x, skip = layer(x)
            skip_sum += skip

        return skip_sum
