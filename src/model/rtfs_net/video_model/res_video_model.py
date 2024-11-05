# This code was borrowed from
# https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/tree/master
import torch
import torch.nn as nn

from .resnet import BasicBlock, ResNet


class ResVideoModel(nn.Module):
    def __init__(
        self,
        relu_type="swish",
    ):
        super(ResVideoModel, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.SiLU()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(
                1,
                self.frontend_nout,
                kernel_size=(5, 7, 7),
                stride=(1, 2, 2),
                padding=(2, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        x = self._threeD_to_2D_tensor(x)
        x = self.trunk(x)
        x = x.view(B, Tnew, x.size(1)).transpose(1, 2).contiguous()

        return x

    def _threeD_to_2D_tensor(self, x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch * s_time, n_channels, sx, sy)
