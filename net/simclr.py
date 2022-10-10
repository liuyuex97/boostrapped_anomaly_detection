class SimCLR(nn.Module):
    def __init__(self, config):
        super(SimCLR, self).__init__()
        self.config = config
        self.alpha = 4096
        self.K = 256
        self.T = config.train.temperature
        self.use_symmetric_logit = config.train.use_symmetric_logit

        self.use_eqco = False
        self.margin = 0.
        if config.train.use_eqco:
            self.use_eqco = True
            self.K = config.train.eqco_k
            self.margin = EqCo(self.T, self.K, self.alpha)

        self.use_dcl = False
        if config.train.use_dcl:
            self.use_dcl = True
            self.tau_plus = config.train.tau_plus
            
        self.use_hcl = False
        if config.train.use_hcl:
            self.use_hcl = True
            self.beta = config.train.beta

        self.ssl_feat_dim = config.model.ssl_feature_dim
        encoder_params = {'norm_layer' : None, 'is_cifar' : 'cifar' in config.dataset.name}
        self.encoder = ResNet_SSL(config.model.arch, config.model.head,
                                  encoder_params=encoder_params, ssl_feat_dim=self.ssl_feat_dim, method='simclr')

    def forward(self, view_1, view_2):
        batch_size = self.config.train.batch_size
        batch_size_this = view_1.shape[0]
        batch_size = view_1.shape[0]
        z_1 = self.encoder(view_1)
        z_2 = self.encoder(view_2)
        """
        (1) symmetric_logit 
            K : 2 * batch_size - 2 per 2 queries
            N : 2 * batch_size 
        (2) normal
            K : batch_size - 1 per 1 query
            N : batch_size
        """
        assert batch_size % batch_size_this == 0

        if self.use_symmetric_logit:
            features = torch.cat([z_1, z_2], dim=0) # [2 * N, 128] # N = 256 per each gpu
            dot_similarities = torch.mm(features, features.t()) # [2 * N, 2 * ]

            pos_ij = torch.diag(dot_similarities, batch_size_this)
            pos_ji = torch.diag(dot_similarities, -batch_size_this)
            pos = torch.cat([pos_ij, pos_ji]).view(2 * batch_size_this, -1)

            diagonal = np.eye(2 * batch_size_this)
            pos_ij_eye = np.eye(2 * batch_size_this, k=batch_size_this)
            pos_ji_eye = np.eye(2 * batch_size_this, k=-batch_size_this)

            neg_mask = torch.from_numpy(1 - (diagonal + pos_ij_eye + pos_ji_eye)).cuda().type(torch.uint8)
            neg = dot_similarities[neg_mask].view(2 * batch_size_this, -1)

            if self.K < 256:
                assert self.use_eqco
                selection_mask = torch.stack([torch.cat([torch.ones(2 * self.K), torch.zeros(neg.shape[1] - 2 *  self.K)])[torch.randperm(neg.shape[1])]
                                              for _ in range(2 * batch_size_this)], dim=0).cuda().type(torch.uint8)
                neg = neg[selection_mask].view(2 * batch_size_this, -1)
        else:
            dot_similarities = torch.mm(z_1, z_2.t()) # [N, N]
            pos = torch.diag(dot_similarities).unsqueeze(-1) # [N, 1]

            diagonal = torch.eye(batch_size)
            neg_mask = (1 - diagonal).cuda().type(torch.uint8)
            neg = dot_similarities[neg_mask].view(batch_size, -1) # [N, N - 1]

            if self.K < 256:
                one_zeros = torch.cat([torch.ones(self.K), torch.zeros(neg.shape[1] - self.K)])
                selection_mask = torch.stack([one_zeros[torch.randperm(neg.shape[1])] for _ in range(batch_size)], dim=0)
                selection_mask = selection_mask.cuda().type(torch.uint8)
                neg = neg[selection_mask].view(2 * batch_size, -1)

        pos_eqco = pos - self.margin
        if self.use_dcl:
            pos_exp = torch.exp(pos / self.T)
            neg_exp = torch.exp(neg / self.T)

            if self.use_hcl:
                importance = torch.exp(self.beta * neg / self.T)
                neg_exp = importance * neg_exp / importance.mean(dim=-1, keepdim=True)

            neg_exp = (-self.tau_plus * pos_exp + neg_exp) / (1 - self.tau_plus)
            neg_exp = torch.clamp(neg_exp, min=np.exp(-1 / self.T))

            pos_eqco_exp = torch.exp(pos_eqco / self.T)
            logits = torch.log(torch.cat([pos_eqco_exp, neg_exp], dim=1))
        else:
            logits = torch.cat([pos_eqco, neg], dim=1)
            logits = logits / self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        logits_original = torch.cat([pos, neg], dim=1)

        return logits, labels, logits_original
