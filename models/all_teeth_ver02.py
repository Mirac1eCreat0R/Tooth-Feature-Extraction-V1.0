import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.sa4 = PointNetSetAbstraction(npoint=512, radius=0.3, nsample=64, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa5 = PointNetSetAbstraction(npoint=128, radius=0.5, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa6 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp4 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp5 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp6 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.conv0 = nn.Conv1d(128, 128, 1)
        self.conv1 = nn.Conv1d(128, 128, 1)
        
        self.drop0 = nn.Dropout(0.5)
        self.drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv1d(256, 512, 1)
        self.drop2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(512, 128, 1)
        self.drop3 = nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(128, 32, 1)
        self.drop4 = nn.Dropout(0.5)
        self.conv5 = nn.Conv1d(32, 1, 1)
        self.drop5 = nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(1, 1, 1)

    def forward(self, xyz):

        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        # set abstraction layers0
        l1_xyz0, l1_points0 = self.sa1(l0_xyz, l0_points)
        l2_xyz0, l2_points0 = self.sa2(l1_xyz0, l1_points0)
        l3_xyz0, l3_points0 = self.sa3(l2_xyz0, l2_points0)

        # feature propagation layers0
        l2_points0 = self.fp3(l2_xyz0, l3_xyz0, l2_points0, l3_points0)
        l1_points0 = self.fp2(l1_xyz0, l2_xyz0, l1_points0, l2_points0)
        l0_points0 = self.fp1(l0_xyz, l1_xyz0, torch.cat([l0_xyz, l0_points],1), l1_points0)

        # set abstraction layers1
        l1_xyz1, l1_points1 = self.sa4(l0_xyz, l0_points)
        l2_xyz1, l2_points1 = self.sa5(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa6(l2_xyz1, l2_points1)

        # feature propagation layers1
        l2_points1 = self.fp4(l2_xyz1, l3_xyz1, l2_points1, l3_points1)
        l1_points1 = self.fp5(l1_xyz1, l2_xyz1, l1_points1, l2_points1)
        l0_points1 = self.fp6(l0_xyz, l1_xyz1, torch.cat([l0_xyz, l0_points],1), l1_points1)

        # FC layers
        feat0 = self.conv0(l0_points0)
        feat1 = self.conv1(l0_points1)
        
        feat0 = self.drop0(feat0)
        feat1 = self.drop1(feat1)

        feat = torch.cat([feat0,feat1],1)

        feat = self.conv2(feat)
        feat = self.drop2(feat)

        feat = self.conv3(feat)
        feat = self.drop3(feat)

        feat = self.conv4(feat)
        feat = self.drop4(feat)

        feat = self.conv5(feat)
        feat = self.drop5(feat)

        feat = self.conv6(feat)

        feat = feat.permute(0, 2, 1)

        return feat


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.mse_loss(pred, target)

        return total_loss