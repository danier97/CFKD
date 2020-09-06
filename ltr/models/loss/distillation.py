import torch.nn as nn
import torch
from torch.nn import functional as F
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d
import math
from pytracking import complex, dcf


class TeacherSoftLoss(nn.Module):
    """
    Loss to match iou outputs of teacher and student ATOMIoUNet
    """
    def __init__(self, reg_loss=nn.MSELoss()):
        super().__init__()
        self.reg_loss = reg_loss

    def forward(self, iou_student, iou_teacher):
        return self.reg_loss(iou_student, iou_teacher)

class AdaptiveHardLoss(nn.Module):
    """
    Loss to match iou outputs of the student ATOMIoUNet with ground truth iou.
    Produce hard loss when student is worse than teacher. Once student gets close to
    teacher, stop the loss to avoid overfitting.
    """
    def __init__(self, reg_loss=nn.MSELoss(), threshold=0.005):
        super().__init__()
        self.reg_loss = reg_loss
        self.threshold = threshold

    def forward(self, iou_student, iou_teacher, iou_gt):
        loss_teacher = self.reg_loss(iou_teacher, iou_gt)
        loss_student = self.reg_loss(iou_student, iou_gt)
        gap = loss_teacher - loss_student
        
        return loss_student if gap < self.threshold else 0.

class TargetResponseLoss(nn.Module):
    """
    Loss to match extracted features, weighted by target location.
    """
    def __init__(self, reg_loss=nn.MSELoss(), match_layers=None, use_w=True):
        super().__init__()
        self.reg_loss = reg_loss
        self.match_layers = match_layers
        self.use_w = use_w
        if match_layers is None:
            self.match_layers = ['layer2', 'layer3']

    def forward(self, ref_feats_s, test_feats_s, ref_feats_t, test_feats_t, target_bb, test_bb=None):
        """
        ref_feats_s  -- dict, reference img feature layers of student extractor
        test_feats_s -- dict, test img feature layers of student extractor
        ref_feats_t  -- dict, reference img feature layers of teacher extractor
        test_feats_t -- dict, test img feature layers of teacher extractor
        target_bb -- Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
        """
        num_sequences = target_bb.shape[1]
        target_bb = target_bb[0,...] # batch x 4
        batch_index = torch.arange(target_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(target_bb.device)
        target_bb = target_bb.clone()
        target_bb[:, 2:4] = target_bb[:, 0:2] + target_bb[:, 2:4]
        roi = torch.cat((batch_index, target_bb), dim=1)

        # find target centers in original image patch
        center_test_orig = None
        if test_bb is not None:
            test_bb = test_bb[0,...] # batch x 4
            center_test_orig = test_bb[:,0:2] + 0.5 * test_bb[:,2:4]

        loss = 0.
        for idx, layer in enumerate(self.match_layers, 5-len(self.match_layers)):
            # calculate scale factor and approx. target patch size, define PrROIPool
            downsample = (1/2)**idx
            patch_sz = math.ceil(58 * downsample)
            patch_sz = patch_sz + 1 if (patch_sz % 2) == 0 else patch_sz

            # prroi = PrRoIPool2D(patch_sz, patch_sz, downsample)

            # retrieve f_s and f_t
            ref_feat_t = ref_feats_t[layer].reshape(-1, num_sequences, *ref_feats_t[layer].shape[-3:])[0,...]
            test_feat_t = test_feats_t[layer] # batch x channel x sz x sz

            ref_feat_s = ref_feats_s[layer].reshape(-1, num_sequences, *ref_feats_s[layer].shape[-3:])[0,...]
            test_feat_s = test_feats_s[layer] # batch x channel x sz x sz

            # get target patch from ref img feat by PrROI pooling
            target_patch_t = prroi_pool2d(ref_feat_t, roi, patch_sz, patch_sz, downsample) # batch x channel x patch_sz x patch_sz 
            target_patch_s = prroi_pool2d(ref_feat_s, roi, patch_sz, patch_sz, downsample) # batch x channel x patch_sz x patch_sz 

            if self.use_w:
                # cross-correlate target patch with test img feat to get weight map
                p = int((patch_sz - 1) / 2)

                batch, cin_t, H, W = test_feat_t.shape
                weight_t = F.conv2d(test_feat_t.view(1, batch*cin_t, H, W), target_patch_t, padding=p, groups=batch)
                weight_t = weight_t.permute([1,0,2,3]) # batch x 1 x sz x sz
                weight_t = weight_t / torch.sum(weight_t)

                batch, cin_s, H, W = test_feat_s.shape
                weight_s = F.conv2d(test_feat_s.view(1, batch*cin_s, H, W), target_patch_s, padding=p, groups=batch)
                weight_s = weight_s.permute([1,0,2,3]) # batch x 1 x sz x sz
                weight_s = weight_s / torch.sum(weight_s)

                # mult weight map with test img feat and stack layers
                test_Q_t = torch.sum(torch.abs(test_feat_t * weight_t), dim=1) # batch x sz x sz
                test_Q_s = torch.sum(torch.abs(test_feat_s * weight_s), dim=1) # batch x sz x sz
            
            else:
                 # Get ground truth Gaussian label for ref and test patch
                feat_sz = torch.Tensor([ref_feat_s.shape[-2], ref_feat_s.shape[-1]]).to(ref_feat_s.device) # Tensor([sz, sz])
                sigma = 0.25 * feat_sz
                center_test = center_test_orig * downsample - 0.5 * feat_sz # origin at center
                g = dcf.label_function_spatial_batch(feat_sz, sigma, center_test) # batch x 1 x sz x sz
                g = g / torch.sum(g)

                test_Q_t = torch.sum(torch.abs(test_feat_t * g), dim=1) # batch x sz x sz
                test_Q_s = torch.sum(torch.abs(test_feat_s * g), dim=1) # batch x sz x sz

            target_Q_t = torch.sum(torch.abs(target_patch_t), dim=1) # batch x patch_sz x patch_sz
            target_Q_s = torch.sum(torch.abs(target_patch_s), dim=1) # batch x patch_sz x patch_sz

            # match target patch feat
            loss += self.reg_loss(target_Q_s, target_Q_t)

            # match test img feat
            loss += self.reg_loss(test_Q_s, test_Q_t)
        
        return loss

class TSKDLoss(nn.Module):
    """
    Objective for distillation.
    Returns TeacherSoftLoss + AdaptiveHardLoss + TargetResponseLoss
    """
    def __init__(self, reg_loss=nn.MSELoss(), w_ts=1., w_ah=0.1, w_tr=100., threshold_ah=0.005, 
                 match_layers=None, use_w=True):
        super().__init__()
        # subcomponent losses, can turn off adaptive hard by setting threshold to None
        self.teacher_soft_loss = TeacherSoftLoss(reg_loss)
        self.adaptive_hard_loss = AdaptiveHardLoss(reg_loss, threshold_ah)
        self.target_response_loss = TargetResponseLoss(reg_loss, match_layers=match_layers, use_w=use_w)
        
        self.w_ts = w_ts
        self.w_ah = w_ah
        self.w_tr = w_tr

    def forward(self, iou, features):
        loss = 0.
        iou_loss = 0.
        if self.w_ts != 0.:
            tmp = self.w_ts * self.teacher_soft_loss(iou['iou_student'], iou['iou_teacher'])
            loss += tmp
            iou_loss += tmp
        if self.w_ah != 0.:
            tmp = self.w_ah * self.adaptive_hard_loss(**iou)
            loss += tmp
            iou_loss += tmp
        if self.w_tr != 0.:
            loss += self.w_tr * self.target_response_loss(**features)
        return loss, iou_loss


class TSsKDLoss(nn.Module):
    """
    Objective for TSsKD distillation.
    """
    def __init__(self, beta=0.5, sigma=0.9, h=0.005, **kwargs):
        super().__init__()
        # subcomponent losses, can turn off adaptive hard by setting threshold to None
        self.loss_TS = TSKDLoss(**kwargs)
        self.reg_loss = nn.SmoothL1Loss()
        
        self.beta = beta
        self.sigma = sigma
        self.h = h

    def forward(self, iou_dull, iou_intel, features_dull, features_intel, epoch):
        loss_TS_dull = self.loss_TS(iou_dull, features_dull)
        loss_TS_intel = self.loss_TS(iou_intel, features_intel)

        loss_SS = self.reg_loss(iou_dull['iou_student'], iou_intel['iou_student'])

        loss_teacher = self.reg_loss(iou_dull['iou_teacher'], iou_dull['iou_gt'])
        loss_dull = self.reg_loss(iou_dull['iou_student'], iou_dull['iou_gt'])
        loss_intel = self.reg_loss(iou_intel['iou_student'], iou_intel['iou_gt'])

        sigma_dull = self.sigma**epoch if loss_dull - loss_teacher < self.h else 0.
        sigma_intel = self.sigma**epoch if loss_intel - loss_teacher < self.h else 0.

        dull = loss_TS_dull + sigma_dull * loss_SS
        intel = loss_TS_intel + self.beta * sigma_intel * loss_SS

        loss = dull + intel

        return loss


class FidelityLoss(nn.Module):
    def __init__(self, reg_loss=nn.MSELoss(), upsample=False, match_layers=None):
        super().__init__()
        self.reg_loss = reg_loss
        self.match_layers = match_layers
        self.upsample = upsample
        if match_layers is None:
            self.match_layers = ['layer2', 'layer3']

        # TODO: fix the second dim for compression loss
        self.upsample_filter = dict()
        self.upsample_filter['layer2'] = torch.empty((128, 32, 1, 1)).cuda()
        self.upsample_filter['layer3'] = torch.empty((256, 64, 1, 1)).cuda()
        if upsample:
            nn.init.xavier_normal_(self.upsample_filter['layer2'])
            nn.init.xavier_normal_(self.upsample_filter['layer3'])

    def forward(self, ref_feats_s, test_feats_s, ref_feats_t, test_feats_t, target_bb, **kwargs):
        """
        ref_feats_s  -- dict, reference img feature layers of student extractor
        test_feats_s -- dict, test img feature layers of student extractor
        ref_feats_t  -- dict, reference img feature layers of teacher extractor
        test_feats_t -- dict, test img feature layers of teacher extractor
        """

        num_sequences = target_bb.shape[1]
        loss = 0.
        for layer in self.match_layers:
            # retrieve f_s and f_t
            ref_feat_t = ref_feats_t[layer].reshape(-1, num_sequences, *ref_feats_t[layer].shape[-3:])[0,...]
            test_feat_t = test_feats_t[layer] # batch x channel x sz x sz

            ref_feat_s = ref_feats_s[layer].reshape(-1, num_sequences, *ref_feats_s[layer].shape[-3:])[0,...]
            test_feat_s = test_feats_s[layer] # batch x channel x sz x sz

            if self.upsample:
                # upsample student features
                ref_feat_s_t = F.conv2d(ref_feat_s, self.upsample_filter[layer])
                test_feat_s_t = F.conv2d(test_feat_s, self.upsample_filter[layer])

                # match ref patch feat
                loss += self.reg_loss(ref_feat_s_t, ref_feat_t)
                # match test patch feat
                loss += self.reg_loss(test_feat_s_t, test_feat_t)
            
            else:
                test_Q_t = torch.sum(torch.abs(test_feat_t), dim=1) # batch x sz x sz
                test_Q_s = torch.sum(torch.abs(test_feat_s), dim=1) # batch x sz x sz

                ref_Q_t = torch.sum(torch.abs(ref_feat_t), dim=1) # batch x sz x sz
                ref_Q_s = torch.sum(torch.abs(ref_feat_s), dim=1) # batch x sz x sz

                # match target patch feat
                loss += self.reg_loss(ref_Q_s, ref_Q_t)
                # match test img feat
                loss += self.reg_loss(test_Q_s, test_Q_t)
        
        return loss


class TrackingLoss(nn.Module):
    """
    Tracking loss, introduced in 
    Wang, N., Zhou, W., Song, Y., Ma, C., & Li, H. (2020). Real-Time Correlation Tracking Via Joint Model Compression and Transfer.
    https://arxiv.org/abs/1907.09831
    """

    def __init__(self, reg_loss=nn.MSELoss(), match_layers=None):
        super().__init__()
        self.reg_loss = reg_loss
        self.match_layers = match_layers
        if match_layers is None:
            self.match_layers = ['layer1', 'layer2', 'layer3']

    def forward(self, ref_feats_s, test_feats_s, target_bb, test_bb, **kwargs):
        """
        ref_feats_s  -- dict, reference img feature layers of student extractor
        test_feats_s -- dict, test img feature layers of student extractor
        target_bb -- Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
        test_bb -- target boxes (x,y,w,h) in test imag. eDims (images, sequences, 4).
        """

        num_sequences = target_bb.shape[1]
        target_bb = target_bb[0,...] # batch x 4
        test_bb = test_bb[0,...] # batch x 4

        # find target centers in original image patch
        center_ref_orig = target_bb[:,0:2] + 0.5 * target_bb[:,2:4]
        center_test_orig = test_bb[:,0:2] + 0.5 * test_bb[:,2:4]

        loss = 0.
        for idx, layer in enumerate(self.match_layers, 5-len(self.match_layers)):
            # calculate scale factor and approx. target patch size, define PrROIPool
            downsample = (1/2)**idx

            # retrieve feat_s
            ref_feat = ref_feats_s[layer].reshape(-1, num_sequences, *ref_feats_s[layer].shape[-3:])[0,...]
            test_feat = test_feats_s[layer] # batch x channel x sz x sz

            # Get ground truth Gaussian label for ref and test patch
            feat_sz = torch.Tensor([ref_feat.shape[-2], ref_feat.shape[-1]]).to(ref_feat.device) # Tensor([sz, sz])
            sigma = 0.25 * feat_sz
            center_ref = center_ref_orig * downsample - 0.5 * feat_sz # origin at center
            center_test = center_test_orig * downsample - 0.5 * feat_sz # origin at center
            y = dcf.label_function_spatial_batch(feat_sz, sigma, center_ref) # batch x 1 x sz x sz
            g = dcf.label_function_spatial_batch(feat_sz, sigma, center_test) # batch x 1 x sz x sz

            # compute DFTs and conjugates
            y_hat_star = complex.conj(torch.rfft(y, 2)) # batch x 1 x sz x sz/2 x 2
            ref_feat_hat = torch.rfft(ref_feat, 2) # batch x channel x sz x sz/2 x 2 
            test_feat_hat = torch.rfft(test_feat, 2) # batch x channel x sz x sz/2 x 2

            # train CF
            top = complex.mult(y_hat_star, ref_feat_hat) # batch x channel x sz x sz/2 x 2
            bot = torch.sum(complex.mult(ref_feat_hat, complex.conj(ref_feat_hat)), dim=1, keepdim=True) # batch x 1 x sz x sz/2 x 2
            w_hat = complex.div(top, bot + 1e-5) # batch x channel x sz x sz/2 x 2

            # compute pred on test feat
            r = torch.irfft(complex.mult(complex.conj(w_hat), test_feat_hat), 2, signal_sizes=g.shape[-2:]) # batch x channel x sz x sz

            # compute regression loss
            loss += self.reg_loss(r, g.expand(-1,r.shape[1],-1,-1))
        
        return loss


class CompressionLoss(nn.Module):
    """
    Objective for compression method (adapted for iou net).
    Returns TeacherSoftLoss + AdaptiveHardLoss + FidelityLoss + TrackingLoss
    """
    def __init__(self, reg_loss=nn.MSELoss(), w_ts=1., w_ah=0.1, w_track=100., w_fd=0.01, threshold_ah=0.005):
        super().__init__()
        # subcomponent losses, can turn off adaptive hard by setting threshold to None
        self.teacher_soft_loss = TeacherSoftLoss(nn.SmoothL1Loss())
        self.adaptive_hard_loss = AdaptiveHardLoss(nn.SmoothL1Loss(), threshold_ah)
        self.fidelity_loss = FidelityLoss(reg_loss, upsample=True)
        self.tracking_loss = TrackingLoss(reg_loss)
        
        self.w_ts = w_ts
        self.w_ah = w_ah
        self.w_track = w_track
        self.w_fd = w_fd

    def forward(self, iou, features):
        loss = 0.
        iou_loss = 0.
        if self.w_ts != 0.:
            tmp = self.w_ts * self.teacher_soft_loss(iou['iou_student'], iou['iou_teacher'])
            loss += tmp
            iou_loss += tmp
        if self.w_ah != 0.:
            tmp = self.w_ah * self.adaptive_hard_loss(**iou)
            loss += tmp
            iou_loss += tmp
        if self.w_fd != 0.:
            loss += self.w_fd * self.fidelity_loss(**features)
        if self.w_track != 0.:
            loss += self.w_track * self.tracking_loss(**features)
        return loss, iou_loss


class CFLoss(nn.Module):
    """
    Loss to match target * search with GT Gaussian map.
    """
    def __init__(self, reg_loss=nn.MSELoss(), match_layers=None):
        super().__init__()
        self.reg_loss = reg_loss
        self.match_layers = match_layers
        if match_layers is None:
            self.match_layers = ['conv1', 'layer1', 'layer2', 'layer3']
        self.downsample = {'conv1': 0.5, 'layer1': 0.5**2, 'layer2': 0.5**3, 'layer3': 0.5**4}

    def forward(self, ref_feats_s, test_feats_s, target_bb, test_bb, **kwargs):
        """
        ref_feats_s  -- dict, reference img feature layers of student extractor
        test_feats_s -- dict, test img feature layers of student extractor
        ref_feats_t  -- dict, reference img feature layers of teacher extractor
        test_feats_t -- dict, test img feature layers of teacher extractor
        target_bb -- Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
        """
        num_sequences = target_bb.shape[1]
        target_bb = target_bb[0,...] # batch x 4
        batch_index = torch.arange(target_bb.shape[0], dtype=torch.float32).reshape(-1, 1).to(target_bb.device)
        target_bb = target_bb.clone()
        target_bb[:, 2:4] = target_bb[:, 0:2] + target_bb[:, 2:4]
        roi = torch.cat((batch_index, target_bb), dim=1)

        # find target centers in original image patch
        test_bb = test_bb[0,...] # batch x 4
        center_test_orig = test_bb[:,0:2] + 0.5 * test_bb[:,2:4]

        loss = 0.
        for layer in self.match_layers:
            # calculate scale factor and approx. target patch size, define PrROIPool
            downsample = self.downsample[layer]
            patch_sz = math.ceil(58 * downsample)
            patch_sz = patch_sz + 1 if (patch_sz % 2) == 0 else patch_sz

            # prroi = PrRoIPool2D(patch_sz, patch_sz, downsample)

            # retrieve f_s
            ref_feat = ref_feats_s[layer].reshape(-1, num_sequences, *ref_feats_s[layer].shape[-3:])[0,...]
            test_feat = test_feats_s[layer] # batch x channel x sz x sz

            # get target patch from ref img feat by PrROI pooling
            target_patch = prroi_pool2d(ref_feat, roi, patch_sz, patch_sz, downsample) # batch x channel x patch_sz x patch_sz 

            # cross-correlate target patch with test img feat to get weight map
            p = int((patch_sz - 1) / 2)

            batch, cin_s, H, W = test_feat.shape
            weight = F.conv2d(test_feat.view(1, batch*cin_s, H, W), target_patch, padding=p, groups=batch)
            weight = weight.permute([1,0,2,3]) # batch x 1 x sz x sz
            weight = weight / torch.max(weight)

            # find GT label
            feat_sz = torch.Tensor([ref_feat.shape[-2], ref_feat.shape[-1]]).to(ref_feat.device) # Tensor([sz, sz])
            sigma = 0.15 * feat_sz
            center_test = center_test_orig * downsample - 0.5 * feat_sz # origin at center
            g = dcf.label_function_spatial_batch(feat_sz, sigma, center_test) # batch x 1 x sz x sz

            # match target patch feat
            loss += self.reg_loss(weight, g)
        
        return loss

class CFKDLoss(nn.Module):
    """
    Objective for proposed compression method (adapted for iou net).
    Returns TeacherSoftLoss + AdaptiveHardLoss + FidelityLoss + CFLoss
    """
    def __init__(self, reg_loss=nn.MSELoss(), w_ts=1., w_ah=0.1, w_cf=1., w_fd=1., threshold_ah=0.005,
                 cf_layers=None, fd_layers=None):
        super().__init__()
        # subcomponent losses, can turn off adaptive hard by setting threshold to None
        self.teacher_soft_loss = TeacherSoftLoss(nn.SmoothL1Loss())
        self.adaptive_hard_loss = AdaptiveHardLoss(nn.SmoothL1Loss(), threshold_ah)
        self.fidelity_loss = FidelityLoss(reg_loss, upsample=False, match_layers=fd_layers)
        self.cf_loss = CFLoss(reg_loss, match_layers=cf_layers)
        
        self.w_ts = w_ts
        self.w_ah = w_ah
        self.w_cf = w_cf
        self.w_fd = w_fd

    def forward(self, iou, features):
        loss = 0.
        iou_loss = 0.
        if self.w_ts != 0.:
            tmp = self.w_ts * self.teacher_soft_loss(iou['iou_student'], iou['iou_teacher'])
            loss += tmp
            iou_loss += tmp
        if self.w_ah != 0.:
            tmp = self.w_ah * self.adaptive_hard_loss(**iou)
            loss += tmp
            iou_loss += tmp
        if self.w_fd != 0.:
            loss += self.w_fd * self.fidelity_loss(**features)
        if self.w_cf != 0.:
            loss += self.w_cf * self.cf_loss(**features)
        return loss, iou_loss
