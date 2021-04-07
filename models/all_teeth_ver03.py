import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        self.sa01 = PointNetSetAbstraction(npoint=4096, radius=0.2, nsample=256, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa02 = PointNetSetAbstraction(npoint=2048, radius=0.4, nsample=256, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa03 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp03 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp02 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp01 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.conv00 = nn.Conv1d(128, 128, 1)
        
        self.drop00 = nn.Dropout(0.5)

        self.sa11 = PointNetSetAbstraction(npoint=2048, radius=0.2, nsample=256, in_channel=128+3, mlp=[64, 64, 128], group_all=False)
        self.sa12 = PointNetSetAbstraction(npoint=1024, radius=0.4, nsample=256, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa13 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fp13 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp12 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp11 = PointNetFeaturePropagation(in_channel=259, mlp=[128, 128, 128])

        self.conv10 = nn.Conv1d(128, 512, 1)
        self.drop10 = nn.Dropout(0.5)
        self.conv11 = nn.Conv1d(512+3, 128, 1)
        self.drop11 = nn.Dropout(0.5)
        self.conv12 = nn.Conv1d(128+3, 32, 1)
        self.drop12 = nn.Dropout(0.5)
        self.conv13 = nn.Conv1d(32+3, 1, 1)
        self.drop13 = nn.Dropout(0.5)
        self.conv14 = nn.Conv1d(1, 1, 1)



    def forward(self,xyz):
        B,C,N = xyz.shape()

        l0_points = xyz
        l0_xyz = xyz
        
        # set abstraction layers0
        l1_xyz0, l1_points0 = self.sa01(l0_xyz, l0_points)
        l2_xyz0, l2_points0 = self.sa02(l1_xyz0, l1_points0)
        l3_xyz0, l3_points0 = self.sa03(l2_xyz0, l2_points0)

        # feature propagation layers0
        l2_points0 = self.fp03(l2_xyz0, l3_xyz0, l2_points0, l3_points0)
        l1_points0 = self.fp02(l1_xyz0, l2_xyz0, l1_points0, l2_points0)
        l0_points0 = self.fp01(l0_xyz, l1_xyz0, torch.cat([l0_xyz, l0_points],1), l1_points0)

        # FC layers
        net0 = self.conv00(l0_points0)
        
        net0 = self.drop00(net0)

        l0_points1 = net0

        l1_xyz, l1_points = self.sa11(l0_xyz, l0_points1)
        l2_xyz, l2_points = self.sa12(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa13(l2_xyz, l2_points)

        l2_points = self.fp13(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp12(l1_xyz, l2_xyz, l1_points, l2_points0)
        l0_points = self.fp11(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points1],1), l1_points)

        net = self.conv10(l0_points)
        net = self.drop10(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv11(net)
        net = self.drop11(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv12(net)
        net = self.drop12(net)
        net = torch.cat([l0_xyz,net],1)
        net = self.conv13(net)
        net = self.drop13(net)
        feat = self.conv14(net)

        feat = feat.permute(0, 2, 1)

        return feat


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss