"""
Author: Raphael Senn <raphaelsenn@gmx.de>
Initial coding: 2025-07-14
"""

import torch
import torch.nn as nn


class Maxout(nn.Module):
    """
    Implementation of a maxout hidden layer.

    Reference:
    Maxout Networks, Goodfellow et al., 2014; https://arxiv.org/abs/1302.4389
    """ 
    def __init__(self,
                 in_features: int,
                 num_units: int, 
                 num_pieces: int 
    ) -> None:
        super().__init__() 

        self.linear = nn.Linear(in_features, num_pieces * num_units)
        self.in_features = in_features
        self.num_pieces = num_pieces
        self.num_units = num_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.linear(x)
        outs = outs.view(x.shape[0], self.num_pieces, self.num_units)
        outs, _ = torch.max(outs, dim=1)
        return outs