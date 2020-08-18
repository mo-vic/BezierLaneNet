import os

import torch
from torch import nn

from models.resnet import resnet18
from utils.utils import load_weight


class CustomResnet(nn.Module):
    def __init__(self, feat_dim, ckpt, max_lane, num_fc_nodes):
        super(CustomResnet, self).__init__()

        self.backbone = resnet18(num_classes=feat_dim)
        if os.path.exists(ckpt):
            load_weight(self.backbone, ckpt, feat_dim)

        self.bn = nn.BatchNorm1d(feat_dim)
        self.activation = nn.LeakyReLU()

        self.fc1 = nn.Linear(feat_dim, max_lane + 1)
        self.fc2 = nn.Linear(feat_dim, num_fc_nodes)

    def forward(self, x):
        feature = self.backbone(x)
        feature = self.activation(self.bn(feature))

        out1 = self.fc1(feature)
        out2 = self.fc2(feature)

        return out1, out2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dim", type=int, default=512,
                        help="The output feature dimension of the backbone ResNet.")
    parser.add_argument("--num_fc_nodes", type=int, default=48,
                        help="Number of the nodes in fc layer for control points.")
    parser.add_argument("--max_lane", type=int, default=4, help="Maximum number of lanes.")
    parser.add_argument("--ckpt", type=str, default='', help="Path to the pre-trained weight.")

    args = parser.parse_args()
    print(args)

    model = CustomResnet(args.feat_dim, args.ckpt, args.max_lane, args.num_fc_nodes)
    x = torch.randn([32, 3, 320, 320])
    out1, out2 = model(x)
