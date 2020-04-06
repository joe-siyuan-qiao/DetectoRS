import math
import torch
import torch.nn.functional as F

from .. import builder
from ..registry import DETECTORS
from .htc import HybridTaskCascade


class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@DETECTORS.register_module
class RecursiveFeaturePyramid(HybridTaskCascade):

    def __init__(self,
                 num_stages,
                 backbone,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 interleaved=True,
                 mask_info_flow=True,
                 rfp_steps=2,
                 rfp_sharing=False,
                 stage_with_rfp=(False, True, True, True),
                 **kwargs):
        self.rfp_steps = rfp_steps
        self.rfp_sharing = rfp_sharing
        self.stage_with_rfp = stage_with_rfp
        backbone["rfp"] = None
        backbone["stage_with_rfp"] = stage_with_rfp
        neck_out_channels = kwargs["neck"]["out_channels"]
        if rfp_sharing:
            backbone["rfp"] = neck_out_channels
        super().__init__(
                 num_stages,
                 backbone,
                 semantic_roi_extractor,
                 semantic_head,
                 semantic_fusion,
                 interleaved,
                 mask_info_flow,
                 **kwargs)
        if not self.rfp_sharing:
            backbone["rfp"] = neck_out_channels
            self.rfp_modules = torch.nn.ModuleList()
            for rfp_idx in range(1, rfp_steps):
                rfp_module = builder.build_backbone(backbone)
                rfp_module.init_weights(kwargs["pretrained"])
                self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(neck_out_channels, neck_out_channels // 4)
        self.rfp_weight = torch.nn.Conv2d(
            neck_out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.rfp_weight.weight.data.fill_(0)
        self.rfp_weight.bias.data.fill_(0)

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.neck(x)
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = tuple(self.rfp_aspp(x[i]) if self.stage_with_rfp[i] else x[i]
                              for i in range(len(self.stage_with_rfp)))
            if self.rfp_sharing:
                x_idx = self.backbone.rfp_forward(img, rfp_feats)
            else:
                x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            x_idx = self.neck(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                x_new.append(add_weight * x_idx[ft_idx] + (1 - add_weight) * x[ft_idx])
            x = x_new
        return x
