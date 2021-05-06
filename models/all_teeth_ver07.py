import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

'''
feature_extract
2021.3.27
分类子网络去了 效果很差 
直接把label当先验输入
缩小网络层次 减小参数量和显存压力
在全连接每次都contact label
v0.7对应网络输入点为4096
只预测一类关键点heatmap
'''

class get_model(nn.Module):
    def __init__(self,num_class):
        super(get_model, self).__init__()

        self.sa01 = PointNetSetAbstraction(npoint=4096, radius=0.2, nsample=64, in_channel=6+num_class, mlp=[64, 64, 128], group_all=False)
        self.sa02 = PointNetSetAbstraction(npoint=1024, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa03 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp03 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp02 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp01 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.conv00 = nn.Conv1d(128+3+num_class, 128, 1)
        self.drop00 = nn.Dropout(0.5)

        self.conv10 = nn.Conv1d(128+3+num_class, 64, 1)
        self.drop10 = nn.Dropout(0.5)
        self.conv12 = nn.Conv1d(64+3+num_class, 1, 1)



    def forward(self,xyz,seg_gt):
        B,C,N = xyz.size()
        l0_points = xyz[:, :, :]
        l0_xyz = xyz[:, :3, :]

        l0_points00 = torch.cat([l0_points,seg_gt],1)
        # set abstraction layers0
        l1_xyz0, l1_points0 = self.sa01(l0_xyz, l0_points00)
        l2_xyz0, l2_points0 = self.sa02(l1_xyz0, l1_points0)
        l3_xyz0, l3_points0 = self.sa03(l2_xyz0, l2_points0)

        # feature propagation layers0
        l2_points0 = self.fp03(l2_xyz0, l3_xyz0, l2_points0, l3_points0)
        l1_points0 = self.fp02(l1_xyz0, l2_xyz0, l1_points0, l2_points0)
        l0_points0 = self.fp01(l0_xyz, l1_xyz0, torch.cat([l0_xyz, l0_points],1), l1_points0)

        # FC layers
        net = torch.cat([l0_xyz,seg_gt,l0_points0],1)
        net = self.conv00(net)
        net = self.drop00(net)

        net = torch.cat([l0_xyz,seg_gt,net],1)
        net = self.conv10(net)
        net = self.drop10(net)

        net = torch.cat([l0_xyz,seg_gt,net],1)

        heatmap1 = self.conv12(net)

        return heatmap1


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, heatmap1, heatmap1_gt):

        lossH1 = F.mse_loss(heatmap1, heatmap1_gt)
        return lossH1

# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, heatmap1, heatmap2, heatmap3, heatmap4, heatmap1_gt, heatmap2_gt, heatmap3_gt, heatmap4_gt):
#         lossH1 = F.mse_loss(heatmap1, heatmap1_gt)
#         lossH2 = F.mse_loss(heatmap2, heatmap2_gt)
#         lossH3 = F.mse_loss(heatmap3, heatmap3_gt)
#         lossH4 = F.mse_loss(heatmap4, heatmap4_gt)

#         return lossH1+lossH2+lossH3+lossH4

if __name__ == '__main__':
    import  torch
    model = get_model(14)
    xyz = torch.rand(2, 3, 4096)
    seg = torch.rand(2, 14, 4096)
    h1 = model(xyz,seg)
    print(xyz.size())