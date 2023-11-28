# ---------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
# from torch.nn import DataParallel  # or your customized DataParallel module
# from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback
# ‘_a’ means all 

from self_trans import SelfTrans
from rendering_trans import RenderTrans
from grounding_trans import GroundTrans
# import nn as mynn
# from dropblock import DropBlock2D

class FPT(nn.Module):
    def __init__(self, feature_dim, with_norm='none', upsample_method='bilinear'): # feature_dim 应该是输出特征图的通道数
        super(FPT, self).__init__()
        self.feature_dim = feature_dim
        assert upsample_method in ['nearest', 'bilinear']
        def interpolate(input):
            return F.interpolate(input, scale_factor=2, mode=upsample_method, align_corners=False if upsample_method == 'bilinear' else None)
        self.fpn_upsample = interpolate # 下采样
        assert with_norm in ['group_norm', 'batch_norm', 'none']
        if with_norm == 'batch_norm':
            norm = nn.BatchNorm2d
        elif with_norm == 'group_norm':
            def group_norm(num_channels):
                return nn.GroupNorm(32, num_channels)
            norm = group_norm
        self.st_p5 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim) # 同层之间的自注意力
        self.st_p4 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        self.st_p3 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        self.st_p2 = SelfTrans(n_head = 1, n_mix = 2, d_model = feature_dim, d_k= feature_dim, d_v= feature_dim)
        
        self.gt_p4_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True) # 自上而下的交互 输出和低级特征图(大特征图)的size一样
        self.gt_p3_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p3_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p3 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p4 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        self.gt_p2_p5 = GroundTrans(in_channels=feature_dim, inter_channels=None, mode='dot', dimension=2, bn_layer=True)
        
        self.rt_p5_p4 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False) # 自下而上的交互 输出和高级特征图(小特征图)的size一样
        self.rt_p5_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p5_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p3 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p4_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        self.rt_p3_p2 = RenderTrans(channels_high=feature_dim, channels_low=feature_dim, upsample=False)
        # drop_block = DropBlock2D(block_size=3, drop_prob=0.2)
        
        if with_norm != 'none':
            self.fpn_p5_1x1 = nn.Sequential(*[nn.Conv2d(2048, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p4_1x1 = nn.Sequential(*[nn.Conv2d(1024, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p3_1x1 = nn.Sequential(*[nn.Conv2d(512, feature_dim, 1, bias=False), norm(feature_dim)])
            self.fpn_p2_1x1 = nn.Sequential(*[nn.Conv2d(256, feature_dim, 1, bias=False), norm(feature_dim)])
            
            self.fpt_p5 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p4 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p3 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
            self.fpt_p2 = nn.Sequential(*[nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1, bias=False), norm(feature_dim)])
        else:
            self.fpn_p5_1x1 = nn.Conv2d(2048, feature_dim, 1)
            self.fpn_p4_1x1 = nn.Conv2d(1024, feature_dim, 1)
            self.fpn_p3_1x1 = nn.Conv2d(512, feature_dim, 1)
            self.fpn_p2_1x1 = nn.Conv2d(256, feature_dim, 1)
            
            self.fpt_p5 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p4 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p3 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)
            self.fpt_p2 = nn.Conv2d(feature_dim*5, feature_dim, 3, padding=1)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, a=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, res2, res3, res4, res5):
        fpn_p5_1 = self.fpn_p5_1x1(res5) # 通道降维 其实也没有降维 降到256 与输出通道相同
        fpn_p4_1 = self.fpn_p4_1x1(res4)
        fpn_p3_1 = self.fpn_p3_1x1(res3)
        fpn_p2_1 = self.fpn_p2_1x1(res2)
        fpt_p5_out = torch.cat((self.st_p5(fpn_p5_1), self.rt_p5_p4(fpn_p5_1, fpn_p4_1), 
            self.rt_p5_p3(fpn_p5_1,fpn_p3_1), self.rt_p5_p2(fpn_p5_1,fpn_p2_1), fpn_p5_1), 1)
        fpt_p4_out = torch.cat((self.st_p4(fpn_p4_1), self.rt_p4_p3(fpn_p4_1, fpn_p3_1), 
            self.rt_p4_p2(fpn_p4_1,fpn_p2_1), self.gt_p4_p5(fpn_p4_1,fpn_p5_1), fpn_p4_1), 1)
        fpt_p3_out = torch.cat((self.st_p3(fpn_p3_1), self.rt_p3_p2(fpn_p3_1, fpn_p2_1), 
            self.gt_p3_p4(fpn_p3_1,fpn_p4_1), self.gt_p3_p5(fpn_p3_1,fpn_p5_1), fpn_p3_1), 1)
        fpt_p2_out = torch.cat((self.st_p2(fpn_p2_1), self.gt_p2_p3(fpn_p2_1, fpn_p3_1), 
            self.gt_p2_p4(fpn_p2_1,fpn_p4_1), self.gt_p2_p5(fpn_p2_1,fpn_p5_1), fpn_p2_1), 1)
        fpt_p5 = self.fpt_p5(fpt_p5_out)
        fpt_p4 = self.fpt_p4(fpt_p4_out)
        fpt_p3 = self.fpt_p3(fpt_p3_out)
        fpt_p2 = self.fpt_p2(fpt_p2_out)
        '''
        fpt_p5 = drop_block(self.fpt_p5(fpt_p5_out))
        fpt_p4 = drop_block(self.fpt_p4(fpt_p4_out))
        fpt_p3 = drop_block(self.fpt_p3(fpt_p3_out))
        fpt_p2 = drop_block(self.fpt_p2(fpt_p2_out))
        '''
        return fpt_p2, fpt_p3, fpt_p4, fpt_p5


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    input1=torch.randn(1,256,256,256).to(device)
    input2=torch.randn(1,512,128,128).to(device)
    input3=torch.randn(1,1024,64,64).to(device)
    input4=torch.randn(1,2048,32,32).to(device)
    model = FPT(256)
    model = model.to(device)
    output=model(input1, input2, input3, input4)
    flops, params = profile(model, inputs=(input1, input2, input3, input4, ))
    print('flops:{}'.format(flops/1e9))
    print('params:{}M'.format(params/1e6))
