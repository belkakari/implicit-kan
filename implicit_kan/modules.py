from math import pi

import torch
import torch.nn as nn


class GaussianFourierFeatureTransform(nn.Module):
    """
    From https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_dim*2, width, height].
    """

    def __init__(self, num_input_channels=2, mapping_dim=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self.mapping_dim = mapping_dim
        self._B = torch.randn((num_input_channels, mapping_dim)) * scale

    def forward(self, x, phase=None):
        batches, channels, width, height = x.shape
        assert channels == self._num_input_channels, "Expected input to have {} channels (got {} channels)".format(
            self._num_input_channels, channels
        )

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self.mapping_dim)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        if phase is not None:
            x = 2 * pi * x + phase
        else:
            x = 2 * pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
