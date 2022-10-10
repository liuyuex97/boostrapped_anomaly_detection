import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import os
import sys
import numpy as np
import math
import time
import shutil, errno
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import faiss
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

import torch
import torch.nn as nn


#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_features(model, dataloader, max_images=10 ** 10, verbose=False):
    features, labels = [], []
    total = 0

    model.eval()

    for index, (img, label) in enumerate(dataloader):

        if total > max_images:
            break

        img, label = img.cuda(), label.cuda()

        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    return np.array(features), np.array(labels)



def knn(model, device, val_loader, criterion, args, writer, epoch=0):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1]

            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)

        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


#### OOD detection ####
def get_roc_sklearn(y_pred, y_true):
    #### shanghai
    auroc = skm.roc_auc_score(y_true, y_pred)
    return auroc
    ####
    # labels = [0] * len(y_true) + [1] * len(y_pred)
    # data = np.concatenate((y_true, y_pred))
    # auroc = skm.roc_auc_score(labels, data)
    # return auroc


def get_pr_sklearn(y_pred, y_true):
    #### shanghai
    aupr = skm.average_precision_score(y_true, y_pred)
    return aupr
    ####
    # labels = [0] * len(y_true) + [1] * len(y_pred)
    # data = np.concatenate((y_true, y_pred))
    # aupr = skm.average_precision_score(labels, data)
    # return aupr

def get_fpr(y_pred, y_true):
    #### shanghai
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # fpr = fp/(fp+tn)
    # return fpr
    ####
    return np.sum(y_pred < np.percentile(y_true, 95)) / len(y_pred)

def get_decision(xin, xood, conf=50):
    # xin: calibration set features
    # xood: query features
    decision_list = (xood > np.percentile(xin, conf)).astype(int)
    return decision_list

def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=True):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    # calculate mahalanobis distance between calibration set and train set
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    # calculate mahalanobis distance between query set and train set
    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = (
        torch.tensor(norm_layer.mean).view(1, 3, 1, 1),
        torch.tensor(norm_layer.std).view(1, 3, 1, 1),
    )
    return x * s + m


class ssdk_dataset(torch.utils.data.Dataset):
    def __init__(self, images, norm_layer, copies=1, s=32):
        self.images = images

        # immitating transformations used at training self-supervised models
        # replace it if training models with a different data augmentation pipeline
        self.tr = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(s, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                norm_layer,
            ]
        )

        self.n = len(images)
        self.size = len(images) * copies

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.tr(self.images[idx % self.n]), 0


def sliceloader(dataloader, norm_layer, k=1, copies=1, batch_size=128, size=32):

    images, labels = readloader(dataloader)
    indices = np.random.permutation(np.arange(len(images)))
    images, labels = images[indices], labels[indices]

    index_k = torch.cat(
        [torch.where(labels == i)[0][0:k] for i in torch.unique(labels)]
    ).numpy()
    index_not_k = np.setdiff1d(np.arange(len(images)), index_k)

    dataset_k = ssdk_dataset(
        unnormalize(images[index_k], norm_layer), norm_layer, copies, size
    )
    dataset_not_k = torch.utils.data.TensorDataset(
        images[index_not_k], labels[index_not_k]
    )
    print(
        f"Number of selected OOD images (k * num_classes_ood_dataset) = {len(index_k)} \nNumber of OOD images after augmentation  = {len(dataset_k)} \nRemaining number of test images in OOD dataset = {len(dataset_not_k)}"
    )

    loader_k = torch.utils.data.DataLoader(
        dataset_k, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    loader_not_k = torch.utils.data.DataLoader(
        dataset_not_k,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader_k, loader_not_k

# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, labelstrain, clusters):
    # import pdb; pdb.set_trace()
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        ypred = get_clusters(ftrain, clusters)
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood

def get_result_dict(name_dict, labelstest, dood):
    for i in range(len(labelstest)):
        # import pdb; pdb.set_trace()
        if labelstest[i] in name_dict:
            if name_dict[labelstest[i]] < dood[i]:
                name_dict[labelstest[i]] = dood[i]
    # for k in name_dict:
    #     name_dict[k] = 0 if np.sum(name_dict[k]) == 0 else 1
    return name_dict

def get_frame_result(name_dict, frame_name):
    result_list = np.zeros((len(frame_name)))
    for i in range(len(frame_name)):
        if frame_name[i] in name_dict:
            result_list[i] = name_dict[frame_name[i]]
    result_list = np.asarray(result_list)
    return result_list

def get_maxpool_result(name_dict, pruned_names, pruned_labels):
    dood = []
    true_label = []
    pruned_names = pruned_names.tolist()
    for name in name_dict:
        idx_true_label = pruned_names.index(name)
        true_label.append(pruned_labels[idx_true_label])
        dood.append(name_dict[name])
    dood = np.asarray(dood)
    true_label = np.asarray(true_label)
    return dood, true_label

def get_skeleton_eval_results(ftrain, ftest, food_known, food, labelstrain, labelstest):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10 # training set
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10 # calibration set
    food_known /= np.linalg.norm(food_known, axis=-1, keepdims=True) + 1e-10 # known out of sample examples
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10 # test samples for detection

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True) # get mean and std of training sample

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food_known = (food_known - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)
    plot_all = False
    # import pdb; pdb.set_trace()
    if plot_all:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import seaborn as sns
        import pandas as pd
        fall = np.concatenate((ftrain, food))
        # import pdb; pdb.set_trace()
        label_all = np.concatenate((labelstrain, labelstest))
        inx_all = np.arange(0, len(fall))
        tsne_idx = np.random.choice(inx_all, int(len(fall)*0.8))
        tsne = TSNE(n_components=2, random_state=0)
        X, y = fall[tsne_idx], label_all[tsne_idx]
        fall_2d = tsne.fit_transform(X, y)
        plt.figure(figsize=(16, 10))
        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = fall_2d[:, 0]
        df_subset['tsne-2d-two'] = fall_2d[:, 1]
        df_subset['y'] = y
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sns.color_palette("hls", 2),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        plt.show()
        dtest, dood = get_scores(ftrain, ftest, food, labelstrain)
        dtest2, dood2 = get_scores(food_known, ftest, food, labelstrain)
        dtest, dood = np.abs(dtest - dtest2), dood - dood2
        fpr95 = get_fpr(dtest, dood)
        import pdb; pdb.set_trace()
        auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
        return fpr95, auroc, aupr
    else:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import seaborn as sns
        import pandas as pd

        # calculate mahalanobis distance of calibration set to train set and query set to train set
        dtest, dood = get_scores(ftrain, ftest, food, labelstrain)
        dtest2, dood2 = get_scores(food_known, ftest, food, labelstrain)
        dtest, dood = np.abs(dtest - dtest2), dood - dood2
        import pdb; pdb.set_trace()
        # get decision list
        decision_ood = get_decision(dtest, dood, conf=95)

        # original evaluations
        accuracy = skm.accuracy_score(decision_ood, labelstest)
        print('accuracy', accuracy)
        fpr95 = get_fpr(decision_ood, labelstest)
        auroc, aupr = get_roc_sklearn(decision_ood, labelstest), get_pr_sklearn(decision_ood, labelstest)
        return fpr95, auroc, aupr

def get_eval_results(ftrain, ftest, food_known, food_not_known, labelstrain, labelstest, numerictest):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10 # training set
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10 # calibration set
    food_known /= np.linalg.norm(food_known, axis=-1, keepdims=True) + 1e-10 # known out of sample examples
    food_not_known /= np.linalg.norm(food_not_known, axis=-1, keepdims=True) + 1e-10 # test samples for detection

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True) # get mean and std of training sample

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food_known = (food_known) / (s + 1e-10)
    food_not_known = (food_not_known - m) / (s + 1e-10)

    # calculate mahalanobis distance of calibration set to train set and query set to train set
    dtest1, dood1 = get_scores(ftrain, ftest, food_not_known, labelstrain, 20)
    dtest2, dood2 = get_scores(food_known, ftest, food_not_known, labelstrain, 20)
    # np.save('ftrain.npy', ftrain)
    # np.save('food_not_known.npy', food_not_known)
    # np.save('labelstrain.npy', labelstrain)
    # np.save('numerictest.npy', numerictest)
    dtest, dood = dtest1 - dtest2, dood1 - dood2

    # create dictionary with unique test labels as keys
    # load label list and corresponding label names
    frame_label = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/pruned_labels.npy')
    frame_name = np.load('/home/liuyuex/Documents/CrosSCLR/data/HR-ShanghaiTech/pruned_name_list.npy', allow_pickle=True)
    name_dict = dict.fromkeys(frame_name, -1e10)

    # get outputs by frame ID and store it in the dictionary
    name_dict = get_result_dict(name_dict, labelstest, dood)
    dood, true_label = get_maxpool_result(name_dict, frame_name, frame_label)
    dood = (dood - np.min(dood)) / (np.max(dood) - np.min(dood))
    ood = dood[np.where(true_label==1)]
    test = dood[np.where(true_label==0)]
    # import pdb; pdb.set_trace()

    # original evaluations
    fpr95 = get_fpr(dood, true_label)
    auroc, aupr = get_roc_sklearn(dood, true_label), get_pr_sklearn(dood, true_label)
    return fpr95, auroc, aupr

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
