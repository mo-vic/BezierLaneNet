import os
import io
import argparse

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.backends import cudnn

from utils.utils import mkdir, eval_bezier
from utils.dataloader import build_dataloader
from models.custom_resnet import CustomResnet

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image.close()
    return img


def visualize_image(writer, images, labels, outputs, gt_num_lanes, pred_num_lanes, degree):
    images = images.clone().cpu().data.numpy()
    outputs = outputs.detach().cpu().data.numpy()
    labels = labels.clone().cpu().data.numpy()
    gt_num_lanes = gt_num_lanes.clone().cpu().data.numpy()
    pred_num_lanes = pred_num_lanes.clone().cpu().data.numpy()

    for image, label, output, gt_num_lane, pred_num_lane in zip(images, labels, outputs, gt_num_lanes, pred_num_lanes):
        rendered_image = plot(image, label, output, gt_num_lane, pred_num_lane, degree)
        writer.write(rendered_image)


def evaluate(model, dataloader, degree, save_name, use_gpu):
    model.eval()

    with torch.no_grad():
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(save_name, fourcc, 30, (820, 295))
            for idx, data in enumerate(tqdm(dataloader, desc="Running inference...")):
                images, existence, coors, ts = data
                if use_gpu:
                    images = images.cuda()
                    existence = existence.cuda()
                    coors = coors.cuda()
                outs1, outs2 = model(images)
                cls_labels = existence.sum(dim=1).long()

                visualize_image(writer, images, coors, outs2, cls_labels, torch.argmax(outs1, dim=1), degree)
        finally:
            writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--degree", type=int, default=5, help="Degree of the bezier curves.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--gpu_ids", type=str, default='', help="Specify the GPU ids.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--input_size", type=int, nargs=2, required=True, help="Size of the input image (w, h).")
    parser.add_argument("--max_lane", type=int, default=4, help="Maximum number of lanes.")
    parser.add_argument("--num_points", type=int, default=72, help="Number of points sampled from each lane.")
    parser.add_argument("--feat_dim", type=int, default=384, help="The output feature dimension of the backbone.")
    parser.add_argument("--save_name", type=str, default="./video/output.avi", help="Path to save the video.")

    args = parser.parse_args()
    print(args)

    for s in args.gpu_ids:
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id: {}".format(s))
            raise ValueError

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu_ids)

    if args.gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
            cudnn.benchmark = True
            torch.cuda.manual_seed(args.seed)
        else:
            use_gpu = False
    else:
        use_gpu = False

    train_loader, val_loader, test_loader, num_fc_nodes = build_dataloader(args.data,
                                                                           args.batch_size,
                                                                           tuple(args.input_size),
                                                                           args.degree,
                                                                           args.num_points,
                                                                           args.max_lane, use_gpu,
                                                                           args.num_workers)

    model = CustomResnet(args.feat_dim, '', args.max_lane, num_fc_nodes)

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt))

    if use_gpu:
        model = model.cuda()

    folder, _ = os.path.split(args.save_name)
    mkdir(folder)

    evaluate(model, test_loader, args.degree, args.save_name, use_gpu)


if __name__ == "__main__":
    main()
