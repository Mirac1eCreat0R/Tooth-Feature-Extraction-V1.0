import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation
from pointnet import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self,num_class):
        super(get_model, self).__init__()
        
        #pointnet sem_seg
        self.k = num_class
        self.feat = PointNetEncoder(global_feat=False, feature_transform=False, channel=3)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


        self.sa01 = PointNetSetAbstraction(npoint=4096, radius=0.2, nsample=256, in_channel=20, mlp=[64, 64, 128], group_all=False)
        self.sa02 = PointNetSetAbstraction(npoint=2048, radius=0.4, nsample=256, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa03 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp03 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp02 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp01 = PointNetFeaturePropagation(in_channel=128+6, mlp=[128, 128, 128])

        self.conv00 = nn.Conv1d(128, 128, 1)
        
        self.drop00 = nn.Dropout(0.5)

        # heatmap1
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
        B,C,N = xyz.size()
        l0_points = xyz[:, :, :]
        l0_xyz = xyz[:, :3, :]

        # pointnet sem_seg
        x, trans, trans_feat = self.feat(l0_xyz)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(B, N, self.k)
        x = x.transpose(2,1)
        
        l0_points00 = torch.cat([l0_points,x],1)
        # set abstraction layers0
        l1_xyz0, l1_points0 = self.sa01(l0_xyz, l0_points00)
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

        # heatmap1
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
        heatmap1 = self.conv14(net)

        return x, heatmap1


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, heatmap1, heatmap1_gt, pred_cls, target_cls):
        lossCls = F.mse_loss(pred_cls, target_cls)
        lossH1 = F.mse_loss(heatmap1, heatmap1_gt)
        return lossCls+lossH1

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
    xyz = torch.rand(2, 3, 8192)
    h1 = model(xyz)
    print(xyz.size())