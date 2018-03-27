import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.autograd.function import once_differentiable
import layers.cppcuda_cffi.roialign as roialign


class RoIAlignFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale, sampling_ratio):
        # ctx.save_for_backward(rois)
        ctx.rois = rois
        ctx.features_size = features.size()
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio

        # compute
        if features.is_cuda != rois.is_cuda:
            raise TypeError('features and rois should be on same device (CPU or GPU)')
        elif features.is_cuda and rois.is_cuda:
            num_channels = features.size(1)
            num_rois = rois.size(0)
            output = torch.zeros(num_rois, num_channels, pooled_height, pooled_width).cuda()
            roialign.roi_align_forward_cuda(features,
                                            rois,
                                            output,
                                            pooled_height,
                                            pooled_width,
                                            spatial_scale,
                                            sampling_ratio)

        elif (not features.is_cuda) and (not rois.is_cuda):
            num_channels = features.size(1)
            num_rois = rois.size(0)
            output = torch.zeros(num_rois, num_channels, pooled_height, pooled_width)
            roialign.roi_align_forward_cpu(features,
                                           rois,
                                           output,
                                           pooled_height,
                                           pooled_width,
                                           spatial_scale,
                                           sampling_ratio)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # rois, = ctx.saved_variables
        rois = ctx.rois
        features_size = ctx.features_size
        pooled_height = ctx.pooled_height
        pooled_width = ctx.pooled_width
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio

        # rois = ctx.rois
        if rois.is_cuda:
            grad_input = torch.zeros(features_size).cuda(rois.get_device())  # <- the problem!
            roialign.roi_align_backward_cuda(rois,
                                             grad_output,
                                             grad_input,
                                             pooled_height,
                                             pooled_width,
                                             spatial_scale,
                                             sampling_ratio)

        else:
            raise NotImplementedError("backward pass not implemented on cpu in cffi extension")
        return grad_input, None, None, None, None, None


class RoIAlign(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, sampling_ratio=0):
        super(RoIAlign, self).__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        # features is a Variable/FloatTensor of size BxCxHxW
        # rois is a (optional: list of) Variable/FloatTensor IDX,Xmin,Ymin,Xmax,Ymax (normalized to [0,1])
        rois = self.preprocess_rois(rois)
        output = RoIAlignFunction.apply(features,
                                        rois,
                                        self.pooled_height,
                                        self.pooled_width,
                                        self.spatial_scale,
                                        self.sampling_ratio)
        return output

    @staticmethod
    def preprocess_rois(rois):
        # do some verifications on what has been passed as rois
        if isinstance(rois, list):  # if list, convert to single tensor (used for multiscale)
            rois = torch.cat(tuple(rois), 0)
        if isinstance(rois, Variable):
            if rois.dim() == 3:
                if rois.size(0) == 1:
                    rois = rois.squeeze(0)
                else:
                    raise ("rois has wrong size")
            if rois.size(1) == 4:
                # add zeros
                zeros = Variable(torch.zeros((rois.size(0), 1)))
                if rois.is_cuda:
                    zeros = zeros.cuda()
                rois = torch.cat((zeros, rois), 1).contiguous()
        return rois