import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import numpy as np

class BYOL(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        super().__init__()

        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.tau = 0.999
        if not self.pretrain:
            self.online_network = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        else:
            self.online_network = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.target_network = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.online_network.fc.weight.shape[1]
                self.online_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.online_network.fc)
                self.target_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                       nn.ReLU(),
                                                       self.target_network.fc)

            for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
                param_target.data.copy_(param_online.data)  # initialize
                param_target.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network(self):
        for param_online, param_target in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_target.data = self.tau * param_target.data + (1 - self.tau) * param_online.data

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, view_2, view_1, im_extreme=None):
        if not self.pretrain:
            return self.online_network(view_1)

        predictions_from_view_1 = self.online_network(view_1)
        predictions_from_view_2 = self.online_network(view_2)

        with torch.no_grad():
            targets_to_view_2 = self.target_network(view_1)
            targets_to_view_1 = self.target_network(view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

        self._update_target_network()
        return loss.mean()
