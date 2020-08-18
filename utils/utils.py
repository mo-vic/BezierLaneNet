import os
import io
from collections import OrderedDict

import torch
from torch.nn.init import normal_, zeros_
from torchvision.utils import make_grid

import math
import bezier
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_weight(model, weight, feat_dim=None):
    loaded_state_dict = torch.load(weight)

    state_dict = OrderedDict()

    classifier = "fc"
    w = "fc.weight"
    b = "fc.bias"

    for k, v in loaded_state_dict.items():
        if not classifier in k:
            state_dict[k] = v
        elif k == w:
            if v.size(0) != feat_dim:
                classifier_weight = torch.empty((feat_dim,) + v.size()[1:], dtype=torch.float32)
                normal_(classifier_weight)
                state_dict[k] = classifier_weight
            else:
                state_dict[k] = v
        elif k == b:
            if v.size(0) != feat_dim:
                classifier_bias = torch.empty((feat_dim,), dtype=torch.float32)
                zeros_(classifier_bias)
                state_dict[k] = classifier_bias
            else:
                state_dict[k] = v
        else:
            state_dict[k] = v
    model.load_state_dict(state_dict)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_bezier(ctrl_coor, degree):
    curve = bezier.Curve(ctrl_coor, degree=degree)
    ts = np.linspace(0.0, 1.0, 256)
    x_eval, y_eval = curve.evaluate_multi(ts)

    return x_eval, y_eval


def plot(image, label, output, gt_num_lane, pred_num_lane, degree):
    ax = plt.gca()
    ax.figure.set_size_inches(8.2, 2.95)
    image = image.transpose([1, 2, 0])
    ax.imshow(image)

    cmap = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    ctrl_points = output.reshape((-1, (degree + 1) * 2))
    for idx in range(pred_num_lane):
        x_ctrls = ctrl_points[idx, 0::2]
        y_ctrls = ctrl_points[idx, 1::2]
        ctrl_point = np.stack([x_ctrls, y_ctrls], axis=1).transpose()
        x_eval, y_eval = eval_bezier(ctrl_point, degree)
        ax.plot(x_eval, y_eval, color=cmap[idx], label="prediction", ls="--")

    label = label[:gt_num_lane, :, :]
    label = label.reshape((-1, 2))
    ax.scatter(label[:, 0], label[:, 1], color="C0", s=10, label="reference")

    if gt_num_lane != 0 or pred_num_lane != 0:
        ax.legend(loc="upper right")

    ax.set_xlim(0, 820)
    ax.set_ylim(0, 295)
    ax.axis("off")

    ax.invert_yaxis()

    with io.BytesIO() as buffer:
        plt.savefig(buffer, bbox_inches="tight")
        plt.close("all")
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
        image = image.resize((820, 295))
        img = np.array(image)
        image.close()
    return img


def visualize_image(writer, images, labels, outputs, gt_num_lanes, pred_num_lanes, degree, global_step):
    num_vis = 25
    images = images.clone().cpu().data.numpy()
    outputs = outputs.detach().cpu().data.numpy()
    labels = labels.clone().cpu().data.numpy()
    gt_num_lanes = gt_num_lanes.clone().cpu().data.numpy()
    pred_num_lanes = pred_num_lanes.clone().cpu().data.numpy()

    images = images[-num_vis:, :, :, :]
    outputs = outputs[-num_vis:, :]
    labels = labels[-num_vis:, :, :, :]
    gt_num_lanes = gt_num_lanes[-num_vis:]
    pred_num_lanes = pred_num_lanes[-num_vis:]

    rendered_images = []
    for image, label, output, gt_num_lane, pred_num_lane in zip(images, labels, outputs, gt_num_lanes, pred_num_lanes):
        rendered_image = plot(image, label, output, gt_num_lane, pred_num_lane, degree)
        rendered_images.append(rendered_image)
    rendered_images = np.stack(rendered_images, axis=0)
    rendered_images = rendered_images.transpose((0, 3, 1, 2))
    rendered_images = torch.tensor(rendered_images)

    grid_image = make_grid(rendered_images.data, int(math.sqrt(num_vis)), range=(0, 255))
    writer.add_image("Vis", grid_image, global_step)


def train(model, dataloader, optimizer, criterion, beta, writer, epoch, degree, use_gpu):
    model.train()

    all_loss = []
    all_cls_loss = []
    all_dsd_loss = []
    all_acc = []

    for idx, data in enumerate(tqdm(dataloader, desc="Training epoch {}...".format(epoch))):
        images, existence, coors, ts = data
        if use_gpu:
            images = images.cuda()
            existence = existence.cuda()
            coors = coors.cuda()
            ts = ts.cuda()
        outs1, outs2 = model(images)
        cls_labels = existence.sum(dim=1).long()
        cls_loss = criterion["xent"](outs1, cls_labels)
        targets = {"existence": existence, "ts": ts, "coors": coors}
        dsd_loss = criterion["dsd"](outs2, targets)

        loss = beta * cls_loss + dsd_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (cls_labels == torch.argmax(outs1, dim=1)).float().mean()

        all_loss.append(loss.item())
        all_cls_loss.append(cls_loss.item())
        all_dsd_loss.append(dsd_loss.item())
        all_acc.append(acc.item())

        writer.add_scalar("train_loss", loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_cls_loss", cls_loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_dsd_loss", dsd_loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("train_acc", acc.item(), global_step=epoch * len(dataloader) + idx)

    writer.add_scalar("loss", np.nanmean(all_loss).item(), global_step=epoch)
    writer.add_scalar("cls_loss", np.nanmean(all_cls_loss).item(), global_step=epoch)
    writer.add_scalar("dsd_loss", np.nanmean(all_dsd_loss).item(), global_step=epoch)
    writer.add_scalar("acc", np.nanmean(all_acc).item(), global_step=epoch)
    visualize_image(writer, images, coors, outs2, cls_labels, torch.argmax(outs1, dim=1), degree, global_step=epoch)


def evaluate(model, dataloader, criterion, beta, scheduler, writer, epoch, degree, weight_dir, use_gpu):
    model.eval()

    all_loss = []
    all_cls_loss = []
    all_dsd_loss = []
    all_acc = []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(dataloader, desc="Evaluating epoch {}...".format(epoch))):
            images, existence, coors, ts = data
            if use_gpu:
                images = images.cuda()
                existence = existence.cuda()
                coors = coors.cuda()
                ts = ts.cuda()
            outs1, outs2 = model(images)
            cls_labels = existence.sum(dim=1).long()
            cls_loss = criterion["xent"](outs1, cls_labels)
            targets = {"existence": existence, "ts": ts, "coors": coors}
            dsd_loss = criterion["dsd"](outs2, targets)

            loss = beta * cls_loss + dsd_loss

            all_loss.append(loss.item())
            all_cls_loss.append(cls_loss.item())
            all_dsd_loss.append(dsd_loss.item())

            acc = (cls_labels == torch.argmax(outs1, dim=1)).float().mean()
            all_acc.append(acc.item())

        loss = np.nanmean(all_loss).item()
        cls_loss = np.nanmean(all_cls_loss).item()
        dsd_loss = np.nanmean(all_dsd_loss).item()
        acc = np.nanmean(all_acc).item()

        writer.add_scalar("loss", loss, global_step=epoch)
        writer.add_scalar("cls_loss", cls_loss, global_step=epoch)
        writer.add_scalar("dsd_loss", dsd_loss, global_step=epoch)
        writer.add_scalar("acc", acc, global_step=epoch)
        visualize_image(writer, images, coors, outs2, cls_labels, torch.argmax(outs1, dim=1), degree, global_step=epoch)

        scheduler.step(np.nanmean(all_loss).item(), epoch=epoch)

        torch.save(model.module.state_dict(), os.path.join(weight_dir, "%04d.pth" % epoch))

    return loss, cls_loss, dsd_loss, acc
