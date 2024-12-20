r"""
Graph and GAN networks in SLAT
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .graphconv import CombUnweighted


class LGCN(nn.Module):
    r"""
    Lightweight GCN which remove nonlinear functions and concatenate the embeddings of each layer:

        (:math:`Z = f_{e}(A, X) = Concat( [X, A_{X}, A_{2X}, ..., A_{KX}])W_{e}`)

    Parameters
    ----------
    K
        layers of LGCN
    """

    def __init__(self, K: Optional[int] = 8):
        super().__init__()
        self.conv1 = CombUnweighted(K=K)

    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(feature, edge_index)
        return x


class LGCN_mlp(nn.Module):
    r"""
    LGCN with MLP

    Parameters
    ----------
    input_size
        input dim
    output_size
        output dim
    K
        LGCN layers
    hidden_size
        hidden size of MLP
    dropout
        dropout ratio
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        K: Optional[int] = 8,
        hidden_size: Optional[int] = 512,
        dropout: Optional[float] = 0.2,
    ):
        super().__init__()
        self.conv1 = CombUnweighted(K=K)
        self.fc1 = nn.Linear(input_size * (K + 1), hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(feature, edge_index)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class WDiscriminator(nn.Module):
    r"""
    WGAN Discriminator

    Parameters
    ----------
    hidden_size
        input dim
    hidden_size2
        hidden dim
    """

    def __init__(self, hidden_size: int, hidden_size2: Optional[int] = 512):
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = nn.Linear(hidden_size2, hidden_size2)
        self.output = nn.Linear(hidden_size2, 1)

    def forward(self, input_embd):
        return self.output(
            F.leaky_relu(
                self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)),
                0.2,
                inplace=True,
            )
        )


class transformation(nn.Module):
    r"""
    Transformation in LGCN

    Parameters
    ----------
    hidden_size
        input dim
    """

    def __init__(self, hidden_size: Optional[int] = 512):
        super().__init__()
        self.trans = nn.Parameter(torch.eye(hidden_size))

    def forward(self, input_embd):
        return input_embd.mm(self.trans)


class notrans(nn.Module):
    r"""
    LGCN without transformation
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_embd: torch.Tensor):
        return input_embd


class ReconDNN(nn.Module):
    r"""
    Data reconstruction network

    Parameters
    ----------
    hidden_size
        input dim
    feature_size
        output size (feature input size)
    hidden_size2
        hidden size
    """

    def __init__(self, hidden_size: int, feature_size: int, hidden_size2: Optional[int] = 512):
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size2)
        self.output = nn.Linear(hidden_size2, feature_size)

    def forward(self, input_embd: torch.Tensor):
        return self.output(F.relu(self.hidden(input_embd)))
