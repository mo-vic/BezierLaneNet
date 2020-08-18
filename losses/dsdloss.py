import argparse

import torch
import numpy as np
from scipy.special import comb


class DSDRandomLoss(torch.nn.Module):
    def __init__(self, degree, max_lane, num_points):
        super(DSDRandomLoss, self).__init__()
        self.degree = degree
        self.num_points = num_points
        self.max_lane = max_lane

        c = np.zeros((1, self.degree + 1), dtype=np.float32)
        for i in range(0, self.degree + 1):
            c[0, i] = comb(self.degree, i)

        self.c = c
        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def forward(self, inputs, targets):
        # Get the device info
        device = inputs.device

        # unpacking the targets
        existence = targets["existence"]
        ts = targets["ts"]
        coors = targets["coors"]
        xs = coors[:, :, :, 0].flatten()
        ys = coors[:, :, :, 1].flatten()

        ts = ts.view((-1, 1)).repeat([1, self.degree + 1])
        c = torch.from_numpy(self.c.copy()).to(device)
        pow1 = torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        pow2 = self.degree - torch.arange(0, self.degree + 1, 1, device=device).view((1, self.degree + 1)).float()
        ts = c * torch.pow(ts, pow1) * torch.pow(1 - ts, pow2)

        x_ctrls = inputs[:, 0::2]
        y_ctrls = inputs[:, 1::2]
        x_ctrls = x_ctrls.view((-1, self.degree + 1))
        y_ctrls = y_ctrls.view((-1, self.degree + 1))
        x_ctrls = x_ctrls.repeat_interleave(self.num_points, dim=0)
        y_ctrls = y_ctrls.repeat_interleave(self.num_points, dim=0)
        decoded_x = (ts * x_ctrls).sum(dim=-1)
        decoded_y = (ts * y_ctrls).sum(dim=-1)

        xs = xs.unsqueeze(1)
        ys = ys.unsqueeze(1)
        decoded_x = decoded_x.unsqueeze(1)
        decoded_y = decoded_y.unsqueeze(1)

        gt = torch.cat([xs, ys], dim=1)
        pred = torch.cat([decoded_x, decoded_y], dim=1)

        loss = self.loss(gt, pred).mean(dim=-1)
        loss = loss.view(-1, self.max_lane, self.num_points).mean(dim=-1)
        loss = loss * existence
        loss = loss.mean()

        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Differentiable Shape Decoder.")
    parser.add_argument("--max_lane", type=int, default=4, help="Maximum number of lanes.")
    parser.add_argument("--degree", type=int, default=5, help="Degree of the bezier curves.")
    parser.add_argument("--num_points", type=int, default=256, help="Number of points for computing the loss.")

    args = parser.parse_args()
    print(args)

    batch_size = 3

    num_fc_nodes = (args.degree + 1) * 2 * args.max_lane
    existence = np.ones((batch_size, args.max_lane)).astype(np.float32)
    ts = np.stack([np.linspace(0.0, 1.0, args.num_points)] * (batch_size * args.max_lane), axis=0)
    ts = ts.reshape(batch_size, args.max_lane, args.num_points)
    inputs = np.random.randint(0, 256, (batch_size, num_fc_nodes))
    coors = np.zeros((batch_size, args.max_lane, args.num_points, 2))

    for batch in range(batch_size):
        inputs_copy = inputs.copy()
        x_ctrls = inputs_copy[:, 0::2]
        y_ctrls = inputs_copy[:, 1::2]
        for lane in range(args.max_lane):
            for t_idx, t in enumerate(ts[batch, lane]):
                x = 0
                y = 0
                for i in range(0, args.degree + 1):
                    c = comb(args.degree, i)
                    x += c * t ** i * (1 - t) ** (args.degree - i) * x_ctrls[batch, lane * (args.degree + 1) + i]
                    y += c * t ** i * (1 - t) ** (args.degree - i) * y_ctrls[batch, lane * (args.degree + 1) + i]
                coors[batch, lane, t_idx, 0] = x
                coors[batch, lane, t_idx, 1] = y

    inputs = torch.from_numpy(inputs).float().cuda()
    existence = torch.from_numpy(existence).float().cuda()
    ts = torch.from_numpy(ts).float().cuda()
    coors = torch.from_numpy(coors).float().cuda()
    targets = {"existence": existence, "ts": ts, "coors": coors}

    criterion = DSDRandomLoss(args.degree, max_lane=args.max_lane, num_points=args.num_points)

    loss = criterion(inputs, targets)
    print("Value of loss:", loss.item())
