#!/usr/bin/env python
# pylint: disable=W0201
import random
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
from .ood_utils import *

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LE_Processor(Processor):
    """
        Processor for Linear Evaluation.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.num_grad_layers = 0
        for name, param in self.model.encoder_q.named_parameters():
            # if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
        # self.num_grad_layers = 2
        if hasattr(self.model, 'encoder_q_motion'):
            for name, param in self.model.encoder_q_motion.named_parameters():
                # if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
            # self.num_grad_layers += 2
        if hasattr(self.model, 'encoder_q_bone'):
            for name, param in self.model.encoder_q_bone.named_parameters():
                # if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
            # self.num_grad_layers += 2
        print(self.model)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == self.num_grad_layers

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))


    def test(self, epoch):
        self.model.eval()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()
        results_file = os.path.join("log_ssd.txt")
        logger.addHandler(logging.FileHandler(results_file, "a"))
        logger.info(self.arg)

        train_loader = self.data_loader['train']
        test_loader = self.data_loader['test']
        train_result_frag = []
        train_label_frag = []
        test_result_frag = []
        test_label_frag = []
        test_numeric_label_frag = []

        for data, label in train_loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # inference
            with torch.no_grad():
                output = self.model(data, view=self.arg.view)
            train_result_frag.append(output.data.cpu().numpy())
            train_label_frag.append(label)

        # randomly sample calibration set
        import random
        num_calibrate = int(len(train_result_frag)*0.9)
        calibrate_result_frag = random.sample(train_result_frag, num_calibrate)
        self.calibrate_result = np.concatenate(calibrate_result_frag)
        for data, label in test_loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # inference
            with torch.no_grad():
                output = self.model(data, view=self.arg.view)
            test_result_frag.append(output.data.cpu().numpy())
            # test_label_frag.append(name)
            test_numeric_label_frag.append(label.cpu().numpy())

        # import pdb; pdb.set_trace()
        self.train_result = np.concatenate(train_result_frag)
        self.train_label = np.concatenate(train_label_frag)
        self.test_result = np.concatenate(test_result_frag).squeeze()
        # self.test_label = np.concatenate(test_label_frag)
        self.test_numeric_label = np.concatenate(test_numeric_label_frag)


        # # randomly select 10% of ood data
        import random
        #### shanghai dataset only
        ood_indices = np.where(self.test_numeric_label == 1)[0]
        num_calibrate = int(len(ood_indices)*0.1)
        ood_indices = random.sample(list(ood_indices), num_calibrate)
        self.ood_known = self.test_result[ood_indices]
        # ood_labels = frame_label[ood_indices]
        # # ood_names = frame_label[ood_indices]
        ####
        # num_calibrate = int(len(self.test_result)*0.9)
        # self.ood_known = np.asarray(random.sample(list(self.test_result), num_calibrate))
        # self.test_numeric_label = np.ones(len(self.test_numeric_label))
        self.train_label = np.zeros(len(self.train_label))
        fpr95, auroc, aupr = get_skeleton_eval_results(
            np.copy(self.train_result),
            np.copy(self.calibrate_result),
            np.copy(self.ood_known),
            np.copy(self.test_result),
            np.copy(self.train_label),
            np.copy(self.test_numeric_label)
        )

        logger.info(
            f"In-data = {self.arg.dataset}, Clusters = {self.arg.clusters}, FPR95 = {fpr95}, AUROC = {auroc}, AUPR = {aupr}"
        )
        self.show_eval_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')

        # direct copy from ssd
        parser.add_argument("--exp-name", type=str, default="temp_eval_ssd")
        parser.add_argument(
            "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
        )
        parser.add_argument("--results-dir", type=str, default="./eval_results")

        parser.add_argument("--arch", type=str, default="resnet50")
        parser.add_argument("--classes", type=int, default=10)
        parser.add_argument("--clusters", type=int, default=10)

        parser.add_argument("--dataset", type=str, default="cifar10")
        parser.add_argument(
            "--data-dir", type=str, default="/data/data_vvikash/fall20/SSD/datasets/"
        )
        parser.add_argument(
            "--data-mode", type=str, choices=("org", "base", "ssl"), default="base"
        )
        parser.add_argument("--normalize", action="store_true", default=False)
        parser.add_argument("--batch-size", type=int, default=256)
        parser.add_argument("--size", type=int, default=32)

        parser.add_argument("--gpu", type=str, default="0")
        parser.add_argument("--ckpt", type=str, help="checkpoint path")
        parser.add_argument("--seed", type=int, default=12345)
        # endregion yapf: enable

        return parser
