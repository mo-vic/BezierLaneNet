import os
import json
import argparse
from datetime import datetime

import torch
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from models.custom_resnet import CustomResnet
from losses.dsdloss import DSDRandomLoss

from utils.dataloader import build_dataloader
from utils.utils import mkdir, train, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--degree", type=int, default=5, help="Degree of the bezier curves.")
    parser.add_argument("--log_dir", type=str, default="runs", help="Path to save the tf event.")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--beta", type=float, default=30, help="Loss balancing factor.")
    parser.add_argument("--weight_dir", type=str, default="weights", help="Folder to save the model weights.")
    parser.add_argument("--pretrained_weight", type=str, required=True, help="Path to the pretrained weight.")
    parser.add_argument("--gpu_ids", type=str, default='', help="Specify the GPU ids.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--input_size", type=int, nargs=2, required=True, help="Size of the input image (w, h).")
    parser.add_argument("--max_lane", type=int, default=4, help="Maximum number of lanes.")
    parser.add_argument("--num_points", type=int, default=72, help="Number of points for computing the loss.")
    parser.add_argument("--feat_dim", type=int, default=384, help="The output feature dimension of the backbone.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum rate.")
    parser.add_argument("--factor", type=float, default=0.5, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=1e-2,
                        help="Threshold for measuring the new optimum, to only focus on significant changes. ")
    parser.add_argument("--eval_freq", type=int, default=1, help="Evaluate frequency.")

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

    logtime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.log_name, logtime)
    train_log = os.path.join(log_dir, "train")
    val_log = os.path.join(log_dir, "val")
    mkdir(train_log)
    mkdir(val_log)
    weight_dir = os.path.join(args.weight_dir, args.log_name, logtime)
    mkdir(weight_dir)

    train_loader, val_loader, test_loader, num_fc_nodes = build_dataloader(args.data,
                                                                           args.batch_size,
                                                                           tuple(args.input_size),
                                                                           args.degree,
                                                                           args.num_points,
                                                                           args.max_lane, use_gpu,
                                                                           args.num_workers)

    model = CustomResnet(args.feat_dim, args.pretrained_weight, args.max_lane, num_fc_nodes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.factor, patience=args.patience,
                                  threshold=args.threshold, verbose=True)

    dsd_loss = DSDRandomLoss(args.degree, args.max_lane, args.num_points)
    xent_loss = torch.nn.CrossEntropyLoss()
    criterion = {"xent": xent_loss, "dsd": dsd_loss}

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    with SummaryWriter(log_dir=train_log) as tr_writer:
        with SummaryWriter(log_dir=val_log) as val_writer:
            js = {"best_epoch": 0, "loss": 1e+12, "cls_loss": 1e+12, "dsd_loss": 1e+12, "acc": 0.0, "seed": args.seed}
            for e in range(args.epochs):
                for i, param_group in enumerate(optimizer.param_groups):
                    learning_rate = float(param_group['lr'])
                    tr_writer.add_scalar("lr of group {}".format(i), learning_rate, global_step=e)

                train(model, train_loader, optimizer, criterion, args.beta, tr_writer, e, args.degree, use_gpu)

                if e % args.eval_freq == 0 or e == args.epochs - 1:
                    val_loss, val_cls_loss, val_dsd_loss, val_acc = evaluate(model, val_loader, criterion, args.beta,
                                                                             scheduler, val_writer, e, args.degree,
                                                                             weight_dir, use_gpu)
                    if val_loss < js["loss"]:
                        js["best_epoch"] = e
                        js["loss"] = val_loss
                        js["cls_loss"] = val_cls_loss
                        js["dsd_loss"] = val_dsd_loss
                        js["acc"] = val_acc

            with open(os.path.join(log_dir, "best_result.json"), 'w') as f:
                json.dump(js, f)


if __name__ == "__main__":
    main()
