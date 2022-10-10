import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import numpy as np

class BYOL(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        ### shanghai
        # self.Bone = [(1, 2), (1, 3), (2, 4), (3, 5),
        #               (6, 8), (8, 10), (7, 9), (9, 11),
        #               (12, 14), (14, 16), (13, 15), (15, 17),
        #               (4, 6), (5, 7), (6, 7), (6, 12), (7, 13), (12, 13)]
        ### NTU-RGBD
        # self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
        #              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
        #              (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        ### openpose
        self.Bone = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                        (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                        (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
        if not self.pretrain:
            self.online_network = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.online_network_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.online_network_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=num_class,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

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
            self.online_network_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.target_network_motion = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args=graph_args,
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.online_network_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)
            self.target_network_bone = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                               hidden_dim=hidden_dim, num_class=feature_dim,
                                               dropout=dropout, graph_args=graph_args,
                                               edge_importance_weighting=edge_importance_weighting,
                                               **kwargs)

            # if mlp:  # hack: brute-force replacement
            #     dim_mlp = self.online_network.fc.weight.shape[1]
            #     self.online_network.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
            #                                            nn.BatchNorm1d(dim_mlp),
            #                                            nn.ReLU(),
            #                                            self.online_network.fc)
            #
            #     self.online_network_motion.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
            #                                                   nn.BatchNorm1d(dim_mlp),
            #                                                   nn.ReLU(),
            #                                                   self.online_network.fc)
            #
            #     self.online_network_bone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
            #                                                 nn.BatchNorm1d(dim_mlp),
            #                                                 nn.ReLU(),
            #                                                 self.online_network.fc)
            self.copyParams(self.online_network, self.target_network)
            self.copyParams(self.online_network_motion, self.target_network_motion)
            self.copyParams(self.online_network_bone, self.target_network_bone)

    def copyParams(self, module_src, module_dest):
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data.copy_(param.data)
                dict_dest[name].requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self, module_src, module_dest):
        """
        Momentum update of the key encoder
        """
        params_src = module_src.named_parameters()
        params_dest = module_dest.named_parameters()

        dict_dest = dict(params_dest)

        for name, param in params_src:
            if name in dict_dest:
                dict_dest[name].data = dict_dest[name].data * self.m + param.data * (1 - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, im_q, im_k=None, im_qm=None, im_qmf=None, view='all', cross=False, topk=1, context=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        lam = np.random.beta(1.0, 1.0)
        if cross:
            return self.cross_training(im_q, im_k, topk, context)
        if self.pretrain:
            im_q_motion = torch.zeros_like(im_q)
            im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

            im_qm_motion = torch.zeros_like(im_qm)
            im_qm_motion[:, :, :-1, :, :] = im_qm[:, :, 1:, :, :] - im_qm[:, :, :-1, :, :]

            im_qmf_motion = torch.zeros_like(im_qmf)
            im_qmf_motion[:, :, :-1, :, :] = im_qmf[:, :, 1:, :, :] - im_qmf[:, :, :-1, :, :]

            im_q_bone = torch.zeros_like(im_q)
            for v1, v2 in self.Bone:
                im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]

            im_qm_bone = torch.zeros_like(im_qm)
            for v1, v2 in self.Bone:
                im_qm_bone[:, :, :, v1 - 1, :] = im_qm[:, :, :, v1 - 1, :] - im_qm[:, :, :, v2 - 1, :]

            im_qmf_bone = torch.zeros_like(im_qmf)
            for v1, v2 in self.Bone:
                im_qmf_bone[:, :, :, v1 - 1, :] = im_qmf[:, :, :, v1 - 1, :] - im_qmf[:, :, :, v2 - 1, :]
        else:
            im_q_motion = torch.zeros_like(im_q)
            im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

            im_q_bone = torch.zeros_like(im_q)
            for v1, v2 in self.Bone:
                im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]


        if not self.pretrain:
            if view == 'joint':
                return self.online_network(im_q)
            elif view == 'motion':
                return self.online_network_motion(im_q_motion)
            elif view == 'bone':
                return self.online_network_bone(im_q_bone)
            elif view == 'all':
                return (self.online_network(im_q) + self.online_network_motion(im_q_motion) + self.online_network_bone(im_q_bone)) / 3.
            else:
                raise ValueError

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_k_bone = torch.zeros_like(im_k)
        for v1, v2 in self.Bone:
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        # compute query features
        q = self.online_network(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        q_motion = self.online_network_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.online_network_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        #######mixup########
        q_mixed = self.online_network(im_qm)
        q_mixed = F.normalize(q_mixed, dim=1)
        q_motion_mixed = self.online_network_motion(im_qm_motion)
        q_motion_mixed = F.normalize(q_motion_mixed, dim=1)
        q_bone_mixed = self.online_network_bone(im_qm_bone)
        q_bone_mixed = F.normalize(q_bone_mixed, dim=1)

        q_mixedf = self.online_network(im_qmf)
        q_mixedf = F.normalize(q_mixedf, dim=1)
        q_motion_mixedf = self.online_network_motion(im_qmf_motion)
        q_motion_mixedf = F.normalize(q_motion_mixedf, dim=1)
        q_bone_mixedf = self.online_network_bone(im_qmf_bone)
        q_bone_mixedf = F.normalize(q_bone_mixedf, dim=1)
        ####################

        # compute key features
        with torch.no_grad():  # no gradient to keys

            k = self.target_network(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_motion = self.target_network_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.target_network_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute loss
        loss = self.regression_loss(q, k)+lam*self.regression_loss(q_mixed, k) + (1-lam)*self.regression_loss(q_mixedf, k)
        loss_motion = self.regression_loss(q_motion, k_motion)+lam*self.regression_loss(q_motion_mixed, k_motion) + (1-lam)*self.regression_loss(q_motion_mixedf, k_motion)
        loss_bone = self.regression_loss(q_bone, k_bone)+lam*self.regression_loss(q_bone_mixed, k_bone)+(1-lam)*self.regression_loss(q_bone_mixedf, k_bone)

        # labels: positive key indicators
        self._momentum_update_key_encoder(self.online_network, self.target_network)  # update the key encoder
        self._momentum_update_key_encoder(self.online_network_motion, self.target_network_motion)  # update the key encoder
        self._momentum_update_key_encoder(self.online_network_bone, self.target_network_bone)  # update the key encoder
        return loss.mean(), loss_motion.mean(), loss_bone.mean()

    def cross_training(self, im_q, im_k, im_qm, im_qmf, topk=1, context=True):
        lam = np.random.beta(1.0, 1.0)

        im_q_motion = torch.zeros_like(im_q)
        im_q_motion[:, :, :-1, :, :] = im_q[:, :, 1:, :, :] - im_q[:, :, :-1, :, :]

        im_k_motion = torch.zeros_like(im_k)
        im_k_motion[:, :, :-1, :, :] = im_k[:, :, 1:, :, :] - im_k[:, :, :-1, :, :]

        im_qm_motion = torch.zeros_like(im_qm)
        im_qm_motion[:, :, :-1, :, :] = im_qm[:, :, 1:, :, :] - im_qm[:, :, :-1, :, :]

        im_qmf_motion = torch.zeros_like(im_qmf)
        im_qmf_motion[:, :, :-1, :, :] = im_qmf[:, :, 1:, :, :] - im_qmf[:, :, :-1, :, :]


        im_q_bone = torch.zeros_like(im_q)
        im_k_bone = torch.zeros_like(im_k)

        for v1, v2 in self.Bone:
            im_q_bone[:, :, :, v1 - 1, :] = im_q[:, :, :, v1 - 1, :] - im_q[:, :, :, v2 - 1, :]
            im_k_bone[:, :, :, v1 - 1, :] = im_k[:, :, :, v1 - 1, :] - im_k[:, :, :, v2 - 1, :]

        im_qm_bone = torch.zeros_like(im_qm)
        for v1, v2 in self.Bone:
            im_qm_bone[:, :, :, v1 - 1, :] = im_qm[:, :, :, v1 - 1, :] - im_qm[:, :, :, v2 - 1, :]

        im_qmf_bone = torch.zeros_like(im_qmf)
        for v1, v2 in self.Bone:
            im_qmf_bone[:, :, :, v1 - 1, :] = im_qmf[:, :, :, v1 - 1, :] - im_qmf[:, :, :, v2 - 1, :]

        q = self.online_network(im_q)
        q = F.normalize(q, dim=1)

        q_motion = self.online_network_motion(im_q_motion)
        q_motion = F.normalize(q_motion, dim=1)

        q_bone = self.online_network_bone(im_q_bone)
        q_bone = F.normalize(q_bone, dim=1)

        #######mixup########
        q_mixed = self.encoder_q(im_qm)
        q_mixed = F.normalize(q_mixed, dim=1)
        q_motion_mixed = self.encoder_q_motion(im_qm_motion)
        q_motion_mixed = F.normalize(q_motion_mixed, dim=1)
        q_bone_mixed = self.encoder_q_bone(im_qm_bone)
        q_bone_mixed = F.normalize(q_bone_mixed, dim=1)

        q_mixedf = self.encoder_q(im_qmf)
        q_mixedf = F.normalize(q_mixedf, dim=1)
        q_motion_mixedf = self.encoder_q_motion(im_qmf_motion)
        q_motion_mixedf = F.normalize(q_motion_mixedf, dim=1)
        q_bone_mixedf = self.encoder_q_bone(im_qmf_bone)
        q_bone_mixedf = F.normalize(q_bone_mixedf, dim=1)
        ####################

        with torch.no_grad():
            self._momentum_update_key_encoder_motion()

            k = self.target_network(im_k)
            k = F.normalize(k, dim=1)

            k_motion = self.target_network_motion(im_k_motion)
            k_motion = F.normalize(k_motion, dim=1)

            k_bone = self.target_network_bone(im_k_bone)
            k_bone = F.normalize(k_bone, dim=1)

        # compute loss
        loss = self.regression_loss(q, k_motion)
        loss += self.regression_loss(q, k_bone)
        loss += self.regression_loss(q, k)
        loss = loss + (self.regression_loss(q_mixed, k)+self.regression_loss(q_mixed, k_motion)+self.regression_loss(q_mixed, k_bone))*lam\
               +(self.regression_loss(q_mixedf, k), self.regression_loss(q_mixedf, k_motion), self.regression_loss(q_mixedf, k_bone))*(1-lam)
        loss_motion = self.regression_loss(q_motion, k)
        loss_motion += self.regression_loss(q_motion, k_bone)
        loss_motion += self.regression_loss(q_motion, k_motion)
        loss_motion = loss + (self.regression_loss(q_motion_mixed, k)+self.regression_loss(q_motion_mixed, k_motion)+self.regression_loss(q_motion_mixed, k_bone))*lam\
               +(self.regression_loss(q_motion_mixedf, k), self.regression_loss(q_motion_mixedf, k_motion), self.regression_loss(q_motion_mixedf, k_bone))*(1-lam)

        loss_bone = self.regression_loss(q_bone, k)
        loss_bone += self.regression_loss(q_bone, k_motion)
        loss_bone += self.regression_loss(q_bone, k_bone)
        loss_bone = loss + (self.regression_loss(q_bone_mixed, k)+self.regression_loss(q_bone_mixed, k_motion)+self.regression_loss(q_bone_mixed, k_bone))*lam\
               +(self.regression_loss(q_bone_mixedf, k), self.regression_loss(q_bone_mixedf, k_motion), self.regression_loss(q_bone_mixedf, k_bone))*(1-lam)
        # labels: positive key indicators
        self._momentum_update_key_encoder(self.online_network, self.target_network)  # update the key encoder
        self._momentum_update_key_encoder(self.online_network_motion, self.target_network_motion)  # update the key encoder
        self._momentum_update_key_encoder(self.online_network_bone, self.target_network_bone)  # update the key encoder

        return loss, loss_motion, loss_bone
