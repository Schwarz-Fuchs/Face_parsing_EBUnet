
from __future__ import absolute_import
from __future__ import print_function
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    def __init__(self, ignore_index=-1, weight=None):
        super(OHEMLoss, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def forward(self, predict, target, min_kept=150000, thresh=0.7):
        # 1. 计算每个像素的原始 Loss [B, H, W] -> 展平为 [N]
        loss = self.criterion(predict, target).view(-1)
        # 2. 建立有效像素掩码 (排除 ignore_index)
        valid_mask = (target.view(-1) != self.ignore_index)
        # 只保留有效像素的 Loss
        loss = loss[valid_mask]
        if loss.numel() == 0:
            return torch.tensor(0.0).to(predict.device)
        # 3. 计算概率并提取 Hard Samples
        # 仅对有效像素计算 Softmax 概率
        # 先切片再计算可以节省计算量和显存
        prob = F.softmax(predict, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, predict.size(1))
        curr_prob = prob[valid_mask]
        # 提取对应真实类别的预测概率
        target_valid = target.view(-1)[valid_mask].unsqueeze(1)
        curr_prob = curr_prob.gather(1, target_valid).squeeze()
        # 4. OHEM 核心逻辑
        hard_mask = curr_prob < thresh
        if hard_mask.sum() < min_kept:
            # 如果硬样本不够多，强制取 Loss 最大的前 min_kept 个
            actual_keep = min(int(min_kept), loss.numel())
            loss, _ = torch.topk(loss, actual_keep)
        else:
            # 否则只计算硬样本的平均 Loss
            loss = loss[hard_mask]
        return loss.mean()

class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, n_classes=19, smooth=1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, pred, target):
        # 1. Convert logits to probabilities along the channel dimension (dim=1)
        # 2. Force casting to float32 to prevent precision overflow/underflow,
        #    especially when using Mixed Precision (FP16) training.
        pred = F.softmax(pred, dim=1).float()
        target = target.float()
        batch_size = pred.shape[0]
        # Initialize an empty tensor to store the Dice score for each sample in the batch
        score = torch.zeros(batch_size, device=pred.device)
        for c in range(self.n_classes):
            # Create a binary mask for the current class 'c'
            # target_c will be 1 where the pixel belongs to class c, and 0 otherwise
            target_c = (target == c).float()

            # Extract the prediction probability map for the current class 'c'
            # Shape: [Batch, H, W]
            pred_c = pred[:, c, ...]

            # Calculate the Intersection: sum of (predicted_prob * ground_truth_mask)
            #vsum over the spatial dimensions (dim 1 and 2, which are H and W)
            intersection = torch.sum(pred_c * target_c, dim=(1, 2))

            # Calculate the Union: sum of predicted probabilities + sum of ground truth pixels
            union = torch.sum(pred_c, dim=(1, 2)) + torch.sum(target_c, dim=(1, 2))

            # Compute the Dice Coefficient for class 'c' and add it to the total score
            # Formula: (2 * intersection + smooth) / (total_predicted + total_gt + smooth)
            score += (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - (score / self.n_classes).mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes=19, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
    def forward(self, pred, target):
        """
        pred: [B, 19, H, W] (模型输出的 logits)
        target: [B, H, W] (类别索引，0-18)
        """
        # 1. 将预测值转为概率分布 (0~1)
        pred = F.softmax(pred, dim=1)
        # 2. 将 target 转为 One-hot 编码: [B, 19, H, W]
        target_one_hot = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        # 3. 计算交集和并集
        intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(pred, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        # 4. 计算 Dice 系数
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        # 返回 1 - Dice 的平均值
        return 1 - dice_score.mean()


"""
Implements the loss enforcing smothness for better structured prediction via consensus used
in the CVPR 2020 paper.
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class StructureConsensuLossFunction(nn.Module):
    def __init__(self, consensus_loss_alpha=10.0,
                 consensus_loss_beta=5.0,
                 max_blobs=50):  # 新增：单图计算的最大 Blob 数量，防止 ID 太多崩溃
        super(StructureConsensuLossFunction, self).__init__()
        self.consensus_loss_alpha = consensus_loss_alpha
        self.consensus_loss_beta = consensus_loss_beta
        self.max_blobs = max_blobs

    def forward(self, logit, blobs, target):
        """
        logit: [B, C, H, W]
        blobs: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        B, C, H, W = logit.size()
        blobs = blobs.squeeze(1)
        target = target.squeeze(1)
        # 预先计算 LogSoftmax 和 Softmax，避免在循环内重复计算
        log_prob = F.log_softmax(logit, dim=1)
        prob = F.softmax(logit, dim=1)
        total_loss_blobs = 0.
        count = 0.
        # 这里的 blobs 是由 get_blobs 产生的全局唯一 ID
        blobs_cc = torch.unique(blobs)

        # 瘦身策略 1：如果零件太多（比如 50 个 ID），随机采样 20 个，防止循环堆积显存
        if len(blobs_cc) > self.max_blobs:
            # 排除背景 ID 0
            valid_ids = blobs_cc[blobs_cc != 0]
            selected_ids = valid_ids[torch.randperm(len(valid_ids))[:self.max_blobs]]
        else:
            selected_ids = blobs_cc[blobs_cc != 0]
        for s in selected_ids:
            # idx_blob: [B, H, W] 的布尔掩码
            idx_blob = (blobs == s)
            if not idx_blob.any():
                continue
            # 找到这个 Blob 对应的类别 (假设 Blob 内部类别一致)
            # 使用 .detach() 断开目标引用，减少内存开销
            label_id = target[idx_blob][0].item()
            # --- 1. Label Match Loss (Alpha 项) ---
            # 瘦身策略 2：直接利用 idx_blob 进行均值计算，避免创建大的 NxCxHxW 张量
            # 这里的 prob_mean 为 [C]，代表整个 Blob 在各个类别的平均预测概率
            # 重点：利用广播机制相乘，不使用 repeat
            blob_pixels_count = idx_blob.sum()
            # [C] = ([B, C, H, W] * [B, 1, H, W]).sum() / count
            prob_blob_mean = (prob * idx_blob.unsqueeze(1)).sum(dim=(0, 2, 3)) / blob_pixels_count
            # 计算 NLL Loss: -log(p_target)
            loss_avg = -torch.log(prob_blob_mean[label_id] + 1e-8)
            # --- 2. Consensus Deviation Loss (Beta 项) ---
            # 瘦身策略 3：KL Divergence 简化计算
            # KL = sum(p * (log(p) - log(q)))
            # q 是平均值 prob_blob_mean
            # 只取 Blob 内部的像素进行计算，减少计算量
            # [N_pixels, C]
            p_pixels = prob.permute(0, 2, 3, 1)[idx_blob]
            log_p_pixels = log_prob.permute(0, 2, 3, 1)[idx_blob]
            # prob_blob_mean_target 不需要 repeat，直接参与减法（广播）
            # log(q) = log(prob_blob_mean + 1e-8)
            log_q = torch.log(prob_blob_mean + 1e-8).unsqueeze(0)  # [1, C]
            # KL 公式：p * (log_p - log_q)
            loss_dev = (p_pixels * (log_p_pixels - log_q)).sum(dim=1).mean()
            total_loss_blobs += (self.consensus_loss_alpha * loss_avg +
                                 self.consensus_loss_beta * loss_dev)
            count += 1.0
        return total_loss_blobs / (count + 1e-8)


class StructureConsensuLossFunction_bk(nn.Module):
    """The structure via consenus loss function.
    The class is initialized with `alpha` and `beta` taht controls the trade-off
    between matching the class label `alpha` and enforcing consenus `beta`.
    The loss takes as input also possibile ways of normailzing each term either normalizing by
    the pixels on a blob (default: idx) or normalizing the blob across all the pixel in the
    image (HW) option reduce_pixel='all'. Keep in mind that is option 'all' induce a side-effect
    in the training related to inducing a variable temperature based on the size of the blob,
    that may underperform  so the default option uses 'idx'; the symmetry option for the
    normalization of the consensus is reduce_pixel_kl='idx' this should be also
    kept like this to normalize for all the pixels in a blob

    Args:
    @consensus_loss_alpha: hyper-param that ensures we match the label class
    @consensus_loss_beta: novel hyper-param enforcing consensus
    @reduce_pixel: how to normalize 1st term of Eq. (8) 'idx' will normalize by size each blob
    'all' will normalize by HxW where HxW is the size of the image; 'idx' is the one used.
    @reduce_pixel_kl: same as above but how to normalize 2nd term of Eq. (8) related to consensu.
    'idx' will normalize by size each blob
    'all' will normalize by HxW where HxW is the size of the image; 'idx' is the one used.
    """
    def __init__(self, consensus_loss_alpha=10.0,
                 consensus_loss_beta=5.0,
                 reduce_pixel='idx',
                 reduce_pixel_kl='idx'):
        super(StructureConsensuLossFunction, self).__init__()
        self.consensus_loss_alpha = consensus_loss_alpha
        self.consensus_loss_beta = consensus_loss_beta
        self.reduce_pixel = reduce_pixel
        self.reduce_pixel_kl = reduce_pixel_kl
        logger.info(f'\nLoss instanciated with\n'
                    f'alpha: {self.consensus_loss_alpha}\n'
                    f'beta: {self.consensus_loss_beta}\n'
                    f'normalization on 1st (label) term: {self.reduce_pixel}\n'
                    f'normalization on 2nd (consensus) term: {self.reduce_pixel_kl}'
                    )

    def structure_via_consensus(self, logit, blobs, target):
        '''
        Main function going over the blobs and enforcing consensus

        Args:
        @logit: is the last output for the convolutional decoder *prior* to
        the softmax normalization
        logit.shape is [batch,channel,size_h,size_w]
        @blobs: this is the masks that contains the connected components to go over each blob and
        enforce
        consensus over them.
        blobs.shape is [batch,1,size_h,size_w]
        @target (y): is the the target label mask
        target.shape is [batch,1,size_h,size_w]
        '''
        total_loss_blobs = 0.
        count = 0.
        n, c, h, w = logit.size()  # batch,channel,size_h,size_w
        blobs = blobs.squeeze(1)
        blobs_cc = torch.unique(blobs, sorted=True, dim=None)
        ## looping over blobs
        for s in blobs_cc:
            idx_blob = blobs == s
            ## Cloning the target labels so we can edit for each blob
            target_blob = target.clone()
            ## Consensus loss in each blob
            loss_pix_avg = self.structure_via_consensus_over_blob(idx_blob, target, logit)
            total_loss_blobs += loss_pix_avg
            count += 1.0
        # normalizing the loss for all blobs
        total_loss_blobs /= count
        return total_loss_blobs

    def structure_via_consensus_over_blob(self, idx_blob, target, logit):
        # --- 修正：确保 target 和 idx_blob 维度匹配 ---
        # 如果 target 是 [B, 1, H, W]，去掉通道维变成 [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)
        # 确保 idx_blob 也是 [B, H, W]
        if idx_blob.dim() == 4:
            idx_blob = idx_blob.squeeze(1)
        # 检查 Blob 里的标签是否唯一
        unique_labels = torch.unique(target[idx_blob])

        if torch.unique(target[idx_blob]).shape[0] != 1:
            print(f"DEBUG: Found problematic blob. Labels: {torch.unique(target[idx_blob])}")
            print(f"DEBUG: Blob mask sum: {idx_blob.sum()}")
            # 如果这个数字非常小（比如 1 或 2），那就是旋转产生的边缘噪点

        if unique_labels.shape[0] != 1:
            # 如果报错，说明你的 get_blobs 把不同类别的像素连在了一起
            raise ValueError(f"Blob contains multiple labels: {unique_labels}. "
                             f"Ensure get_blobs separates different classes.")
        ###################################
        ## 1. Computing the average part
        ###################################
        n, c, h, w = logit.size()
        ## building the labels
        ## getting the 1st index in the blob
        ## assumes all pixels in blob have SAME label
        label_id = target[idx_blob][0]
        target_blob_t = label_id.repeat(n).long()
        # Computing probs
        prob = F.softmax(logit, dim=1)  # NxCxHxW
        ## building the average
        idx_blob_t = idx_blob.unsqueeze(1).repeat(1, c, 1, 1)
        ## zeroing the probs not relevant to the blob
        prob_blob = prob * idx_blob_t.float()  # NxCxHxW
        ## here we sum up over the 2D dimensions to see if
        ## there are some CC not used in a batch sample
        ## for example we may have a batch that does not
        ## contain the HAIR class so the sum below will
        ## be zero in this case.
        ## We keep track of those cases in zero_mask variable
        support_logit = idx_blob_t.sum(dim=(2, 3))  # NxC
        zero_mask = (support_logit == 0)  # NxC
        ###############################################
        ## Normalization over the pixels for this blob
        ###############################################
        if self.reduce_pixel != 'all':
            # we normalize the sum only on selected index that belongs to the blob
            # in theory with this all the blobs have the same weight
            # respect to each other (i.e. independent of blob size)
            prob_blob_mean, _ = self.custom_div(prob_blob.sum(dim=(2, 3)),
                                                support_logit.float()  # NxC
                                                )
        else:
            # we normalize over all the pixel mask (large objects are weighted more in this way)
            prob_blob_mean = prob_blob.sum(dim=(2, 3)) / (h * w)
        ## Computing NLL (negative loglikehood)
        loss_avg = torch.nn.functional.nll_loss(torch.log(prob_blob_mean),
                                                target_blob_t,
                                                reduction='none')
        invalid_samples = zero_mask.any(dim=1)
        # loss is zero for samples without the class
        loss_avg[invalid_samples] = 0.0
        loss_avg = loss_avg.mean()

        ###################################
        ## 2. Computing the deviation part
        ###################################
        nozero_log_mask = (prob_blob != 0)  # select the pixel inside the blob, for which prob!=0
        zero_log_mask = (prob_blob == 0)
        log_prob_blob = prob_blob.clone()
        log_prob_blob[nozero_log_mask] = torch.log(prob_blob[nozero_log_mask])  # NxCxHxW
        prob_blob_mean_target = prob_blob_mean.unsqueeze(dim=(2)).unsqueeze(dim=(3)).repeat(1, 1, h, w)  # NxCxHxW
        ## the kl loss is sum_i t_i*[log(t_i)-log(x_i)], i \in C
        prob_blob_mean_target[zero_log_mask] = 1.0  # log(1)=0
        log_prob_blob[zero_log_mask] = 0.0
        loss_dev = torch.nn.functional.kl_div(log_prob_blob,
                                              prob_blob_mean_target,
                                              reduction='none')  # NxCxHxW
        # sum over classes (kl div) then average over NxHxW
        if self.reduce_pixel_kl == 'all':
            loss_dev = torch.mean(torch.sum(loss_dev, dim=1))
        else:
            loss_dev = torch.sum(loss_dev) / nozero_log_mask.float().sum()
        ## Combining the final loss
        final_loss = self.consensus_loss_alpha * loss_avg + self.consensus_loss_beta * loss_dev
        return final_loss

    def custom_div(self, num, den):
        x = den.clone()
        one_mask = (den != 0)  # => one_mask is [0, 1]
        x[one_mask] = num[one_mask] / den[one_mask]
        y = (x)
        zero_mask = (den == 0)  # => zero_mask is [1, 0]
        y[zero_mask] = 0  # => y is [0, 1]
        return y, zero_mask

    def forward(self, logit, blobs, target):
        return self.structure_via_consensus(logit, blobs, target)

