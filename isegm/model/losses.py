import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isegm.utils import misc
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

class NormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(NormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        # label=0, 1-p，label=1, p
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
        # label=0, p，label=1, 1-p
        beta = (1 - pt) ** self._gamma
        # 320*720
        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        # compute p(M)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        # compute 1/p(M)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            # Calculate the average loss over all pixels
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)


class FocalLoss(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0,
                 ignore_label=-1):
        super(FocalLoss, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def forward(self, pred, label, sample_weight=None):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label

        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, self._alpha * sample_weight, (1 - self._alpha) * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))

        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            tsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(label.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (tsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return self._scale * loss


class SoftIoU(nn.Module):
    def __init__(self, from_sigmoid=False, ignore_label=-1):
        super().__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label

        if not self._from_sigmoid:
            pred = torch.sigmoid(pred)

        loss = 1.0 - torch.sum(pred * label * sample_weight, dim=(1, 2, 3)) \
            / (torch.sum(torch.max(pred, label) * sample_weight, dim=(1, 2, 3)) + 1e-8)

        return loss


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

    def forward(self, pred, label):
        label = label.view(pred.size())
        sample_weight = label != self._ignore_label
        label = torch.where(sample_weight, label, torch.zeros_like(label))

        if not self._from_sigmoid:
            loss = torch.relu(pred) - pred * label + F.softplus(-torch.abs(pred))
        else:
            eps = 1e-12
            loss = -(torch.log(pred + eps) * label
                     + torch.log(1. - pred + eps) * (1. - label))

        loss = self._weight * (loss * sample_weight)
        return torch.mean(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

class MutiBceLoss(nn.Module):
    def __init__(self):
        super(MutiBceLoss, self).__init__()


    def forward(self, pred, label):
        bce_loss = nn.BCELoss(size_average=True)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss += bce_loss(di, label)

        return loss

# Multiple Normalized Focal Loss for deep supervision
class MutiNFLLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(MutiNFLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        NFL_loss = NormalizedFocalLossSigmoid(alpha=self.alpha, gamma=self.gamma)
        loss = 0
        for di in pred:
            # di = F.sigmoid(di)
            loss += NFL_loss(di, label)

        return loss

# weighted cross-entropy loss
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        loss = 0.0
        pred = F.sigmoid(pred)
        batch_size = pred.size(0)
        for index in range(batch_size):
            sub_pred = pred[index]
            sub_label = label[index]
            pos_weight = torch.sum(sub_label == 0) / torch.sum(sub_label == 1)
            eps = 1e-12
            sub_loss = -(torch.log(sub_pred[0] + eps) * sub_label[0] * pos_weight
                     + torch.log(1. - sub_pred[0] + eps) * (1. - sub_label[0]))
            loss += torch.mean(sub_loss)
        return loss / batch_size


    @staticmethod
    def _class_weights(input_):
        # normalize the input_ first
        flattened = flatten(input_)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = (nominator / denominator).detach()
        return class_weights


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class MutilWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MutilWeightedCrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        weightedCrossEntropyLoss = WeightedCrossEntropyLoss()
        loss = 0
        for di in pred:
            loss += weightedCrossEntropyLoss(di, label)

        return loss


class AdaptableNormalizedFocalLossSigmoid(nn.Module):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, max_mult=-1, eps=1e-12,
                 from_sigmoid=False, detach_delimeter=True,
                 batch_axis=0, weight=None, size_average=True,
                 ignore_label=-1):
        super(AdaptableNormalizedFocalLossSigmoid, self).__init__()
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label
        self._weight = weight if weight is not None else 1.0
        self._batch_axis = batch_axis

        self._from_logits = from_sigmoid
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._max_mult = max_mult
        self._k_sum = 0
        self._m_max = 0

    def forward(self, pred, label):
        one_hot = label > 0.5
        sample_weight = label != self._ignore_label
        pos_weight = torch.sum(label == 0) / (torch.sum(label == 1) + torch.sum(label == 0))
        neg_weight = 1. - pos_weight
        if not self._from_logits:
            pred = torch.sigmoid(pred)

        alpha = torch.where(one_hot, pos_weight * sample_weight, neg_weight * sample_weight)
        pt = torch.where(sample_weight, 1.0 - torch.abs(label - pred), torch.ones_like(pred))
        beta = (1 - pt) ** self._gamma
        sw_sum = torch.sum(sample_weight, dim=(-2, -1), keepdim=True)
        beta_sum = torch.sum(beta, dim=(-2, -1), keepdim=True)
        mult = sw_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = beta * mult
        if self._max_mult > 0:
            beta = torch.clamp_max(beta, self._max_mult)

        with torch.no_grad():
            ignore_area = torch.sum(label == self._ignore_label, dim=tuple(range(1, label.dim()))).cpu().numpy()
            sample_mult = torch.mean(mult, dim=tuple(range(1, mult.dim()))).cpu().numpy()
            if np.any(ignore_area == 0):
                self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

                beta_pmax, _ = torch.flatten(beta, start_dim=1).max(dim=1)
                beta_pmax = beta_pmax.mean().item()
                self._m_max = 0.8 * self._m_max + 0.2 * beta_pmax

        loss = -alpha * beta * torch.log(torch.min(pt + self._eps, torch.ones(1, dtype=torch.float).to(pt.device)))
        loss = self._weight * (loss * sample_weight)

        if self._size_average:
            bsum = torch.sum(sample_weight, dim=misc.get_dims_with_exclusion(sample_weight.dim(), self._batch_axis))
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis)) / (bsum + self._eps)
        else:
            loss = torch.sum(loss, dim=misc.get_dims_with_exclusion(loss.dim(), self._batch_axis))

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)
        sw.add_scalar(tag=name + '_m', value=self._m_max, global_step=global_step)



#
class MutiAdaptableNFLLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(MutiAdaptableNFLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        ANFL_loss = AdaptableNormalizedFocalLossSigmoid(alpha=self.alpha, gamma=self.gamma)
        loss = 0
        for di in pred:
            # di = F.sigmoid(di)
            loss += ANFL_loss(di, label)

        return loss


# 鲁棒性的T-Loss
# Copyright 2023 University of Basel and Lucerne University of Applied Sciences and Arts Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

class TLoss(nn.Module):
    def __init__(
            self,
            config,
            nu: float = 1.0,
            epsilon: float = 1e-8,
            reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.config = config
        self.D = torch.tensor(
            (self.config.data.w * self.config.data.h),
            dtype=torch.float,
            device=config.device,
        )

        self.lambdas = torch.ones(
            (self.config.data.h, self.config.data.w),
            dtype=torch.float,
            device=config.device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=config.device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=config.device)
        self.reduction = reduction

    def forward(
            self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x H x W).
            target_tensor (torch.Tensor): Ground truth, size (B x H x W).

        Returns:
            torch.Tensor: Total loss value.
        """
        # size (B x H x W)
        input_tensor = input_tensor[:, 0, ...]
        target_tensor = target_tensor[:, 0, ...]

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
                first_term
                + second_term
                + third_term
                + fourth_term
                + fifth_term
                + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )



# This repo holds code for Boundary Difference Over Union Loss For Medical Image Segmentation(MICCAI 2023).
# boundary DoU loss
class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes, gpu_id):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes
        self.gpu_id = gpu_id
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2] + 2, target.shape[-1] + 2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda(int(self.gpu_id))
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(int(self.gpu_id)),
                                      padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target):
        # input : NxCxHxW target: NxHxW
        # inputs = torch.softmax(inputs, dim=1) 源码 多分类
        inputs = F.sigmoid(inputs)
        # target = self._one_hot_encoder(target) 二分类不需要one-hot编码

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())

        loss = 0.0
        # for i in range(0, self.n_classes):
        #     loss += self._adaptive_size(inputs[:, i], target[:, i])  源码多分类
        loss += self._adaptive_size(inputs[:, 0], target[:, 0])
        # return loss / self.n_classes  源码
        return loss


class MutiBoundaryDoULoss(nn.Module):
    def __init__(self,gpu_id):
        super(MutiBoundaryDoULoss, self).__init__()
        self.gpu_id = gpu_id
    def forward(self, pred, label):
        BD_loss = BoundaryDoULoss(n_classes=1,gpu_id=self.gpu_id)
        loss = 0
        for di in pred:
            # di = F.sigmoid(di)
            loss = loss + 0.8 * BD_loss(di, label[0])
            loss = loss + 0.2 * BD_loss(di, label[1])

        return loss

class MutiTLoss(nn.Module):
    def __init__(self,
            config,
            nu: float = 1.0,
            epsilon: float = 1e-8,
            reduction: str = "mean",):
        super(MutiTLoss, self).__init__()
        self.config = config
        self.nu = nu
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, label):
        T_loss = TLoss(config=self.config, nu=self.nu, epsilon=self.epsilon, reduction=self.reduction)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss += T_loss(di, label)

        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1e-5
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss





class MutiDiceLoss(nn.Module):
    def __init__(self,):
        super(MutiDiceLoss, self).__init__()

    def forward(self, pred, label):
        dice_loss = BinaryDiceLoss()
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss = loss + 1.0 * dice_loss(di, label[0])
            # loss = loss + 0.2 * dice_loss(di, label[1])

        return loss



class MutiSoftIoULoss(nn.Module):
    def __init__(self,):
        super(MutiSoftIoULoss, self).__init__()

    def forward(self, pred, label):
        iou_loss = SoftIoU(from_sigmoid=True, ignore_label=-1)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss = loss + 1.0 * iou_loss(di, label[0])
            # loss = loss + 0.2 * iou_loss(di, label[1])

        return loss




class MutiIoUBCELoss(nn.Module):
    def __init__(self,):
        super(MutiIoUBCELoss, self).__init__()

    def forward(self, pred, label):
        iou_loss = SoftIoU(from_sigmoid=True, ignore_label=-1)
        bce_loss = nn.BCELoss(size_average=True)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss = loss + 0.5 * iou_loss(di, label[0]) + 0.5 * bce_loss(di, label[0])

        return loss


class MutiIoUNFLLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(MutiIoUNFLLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, label):
        iou_loss = SoftIoU(from_sigmoid=True, ignore_label=-1)
        nfl_loss = NormalizedFocalLossSigmoid(alpha=self.alpha, gamma=self.gamma, from_sigmoid=True)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss = loss + 0.5 * iou_loss(di, label[0]) + 0.5 * nfl_loss(di, label[0])

        return loss



class MutiDiceBCELoss(nn.Module):
    def __init__(self,):
        super(MutiDiceBCELoss, self).__init__()

    def forward(self, pred, label):
        dice_loss = BinaryDiceLoss()
        bce_loss = nn.BCELoss(size_average=True)
        loss = 0
        for di in pred:
            di = F.sigmoid(di)
            loss = loss + 0.5 * dice_loss(di, label) + 0.5 * bce_loss(di, label)

        return loss



# BoundaryLoss
class BoundaryLoss(nn.Module):
    # def __init__(self, **kwargs):
    def __init__(self, classes) -> None:
        super().__init__()
        # # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        # self.idc: List[int] = kwargs["idc"]
        self.idx = [i for i in range(classes)]

    def compute_sdf1_1(self, img_gt, out_shape):
        """
        compute the normalized signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                -inf|x-y|; x in segmentation
                +inf|x-y|; x out of segmentation
        normalize sdf to [-1, 1]
        """
        img_gt = img_gt.cpu().numpy()
        img_gt = img_gt.astype(np.uint8)

        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
                # ignore background
            for c in range(0, out_shape[1]):
                posmask = img_gt[b].astype(np.bool_)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf

        return normalized_sdf

    def forward(self, outputs, gt):
        """
        compute boundary loss for binary segmentation
        input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
            gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
        output: boundary_loss; sclar
        """
        # outputs_soft = F.softmax(outputs, dim=1)
        outputs_sigm = F.sigmoid(outputs)
        gt = gt[:,0,...]
        gt_sdf = self.compute_sdf1_1(gt, outputs_sigm.shape)
        pc = outputs_sigm[:,0,...]
        dc = torch.from_numpy(gt_sdf[:,0,...]).cuda(1)
        multipled = torch.einsum('bxy, bxy->bxy', pc, dc)
        bd_loss = multipled.mean()

        return bd_loss






class MutiBDLoss(nn.Module):
    def __init__(self,):
        super(MutiBDLoss, self).__init__()

    def forward(self, pred, label):
        bd_loss = BoundaryLoss(classes=0)
        loss = 0
        for di in pred:
            # di = F.sigmoid(di)
            loss = loss + bd_loss(di, label)

        return loss




class TverskyLoss(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        self.classes = classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_true, alpha=0.7, beta=0.3):

        # y_pred = torch.softmax(y_pred, dim=1)
        y_pred = F.sigmoid(y_pred)
        # y_true = self._one_hot_encoder(y_true)
        loss = 0
        for i in range(0, self.classes):
            p0 = y_pred[:, i, :, :]
            ones = torch.ones_like(p0)
            #p1: prob that the pixel is of class 0
            p1 = ones - p0
            g0 = y_true[:, i, :, :]
            g1 = ones - g0
            #terms in the Tversky loss function combined with weights
            tp = torch.sum(p0 * g0)
            fp = alpha * torch.sum(p0 * g1)
            fn = beta * torch.sum(p1 * g0)
            #add to the denominator a small epsilon to prevent the value from being undefined
            EPS = 1e-5
            num = tp
            den = tp + fp + fn + EPS
            result = num / den
            loss += result
        return 1 - loss



class MutiTverskyLoss(nn.Module):
    def __init__(self,):
        super(MutiTverskyLoss, self).__init__()

    def forward(self, pred, label):
        tversky_loss = TverskyLoss(classes=1)
        loss = 0
        for di in pred:
            # di = F.sigmoid(di)
            loss = loss + 1.0 *  tversky_loss(di, label[0])
            # loss = loss + 0.2 *  tversky_loss(di, label[1])

        return loss