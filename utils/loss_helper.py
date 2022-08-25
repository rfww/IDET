import os
import pdb
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from torch.autograd import Variable
# from lib.utils.tools.logger import Logger as Log


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=2, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()

        # Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                # ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)



# "loss": {
#   "loss_type": "fs_auxce_loss",
#   "params": {
#     "ce_reduction": "elementwise_mean",
#     "ce_ignore_index": -1,
#     "ohem_minkeep": 100000,
#     "ohem_thresh": 0.9
#   }
# },

# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSCELoss, self).__init__()
        self.configer = configer
        weight = None
        # if self.configer.exists('loss', 'params') and 'ce_weight' in self.configer.get('loss', 'params'):
        #     weight = self.configer.get('loss', 'params')['ce_weight']
        #     weight = torch.FloatTensor(weight).cuda()

        reduction = 'elementwise_mean'
        # if self.configer.exists('loss', 'params') and 'ce_reduction' in self.configer.get('loss', 'params'):
        #     reduction = self.configer.get('loss', 'params')['ce_reduction']

        # ignore_index = -1
        # if self.configer.exists('loss', 'params') and 'ce_ignore_index' in self.configer.get('loss', 'params'):
        #     ignore_index = self.configer.get('loss', 'params')['ce_ignore_index']

        # self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=None, reduction=reduction)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                # weights = [1.0] * len(inputs)
                weights = 1.0

            # for i in range(len(inputs)):
                # if len(targets) > 1:
                    # target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                loss += weights * self.ce_loss(inputs, targets[:, 0])
                # else:
                    # target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                # loss += weights * self.ce_loss(inputs, targets)

        else:
            # target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, targets[:, 0])

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()



class FSAuxCELoss(nn.Module):
    def __init__(self, configer=None):
        super(FSAuxCELoss, self).__init__()
        self.configer = configer
        self.ce_loss = FSCELoss(self.configer)
        self.bn_aux = ProjectionHead(dim_in=2)
    def forward(self, inputs, targets):
        seg_out = inputs
        aux_out = self.bn_aux(inputs)
        seg_loss = self.ce_loss(seg_out, targets)
        aux_loss = self.ce_loss(aux_out, targets)
        # loss = self.configer.get('network', 'loss_weights')['seg_loss'] * seg_loss
        loss = 1.0 * seg_loss
        # loss = loss + self.configer.get('network', 'loss_weights')['aux_loss'] * aux_loss
        loss = loss + 0.4 * aux_loss
        return loss


#"network":{
#   "backbone": "deepbase_resnet101_dilated8",
#   "model_name": "asp_ocnet",
#   "bn_type": "torchsyncbn",
#   "stride": 8,
#   "factors": [[8, 8]],
#   "loss_weights": {
#     "aux_loss": 0.4,
#     "seg_loss": 1.0
#   }
# },
