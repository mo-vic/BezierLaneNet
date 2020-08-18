import os
import argparse

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class CULane(Dataset):
    def __init__(self, root, input_size, degree, num=72, max_lane=4, split="train"):
        super(CULane, self).__init__()

        assert split in ["train", "val", "test"]
        assert os.path.exists(root)

        self.num = num
        self.split = split
        self.degree = degree
        self.max_lane = max_lane
        self.input_size = input_size

        split_txt_file = os.path.join(root, "list", split + ".txt")
        assert os.path.exists(split_txt_file)

        image_files = []
        with open(split_txt_file, 'r') as f:
            for line in f:
                image_files.append(os.path.join(root, line.strip()[1:]))
        self.image_files = image_files

        self.transform = ToTensor()

    def __getitem__(self, item):
        image_file = self.image_files[item]
        text_file = image_file.replace(".jpg", ".lines.txt")
        assert os.path.exists(text_file)

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        target_w, target_h = self.input_size
        image = cv2.resize(image, self.input_size)
        image = self.transform(image)

        ratio_w = target_w / w
        ratio_h = target_h / h
        ratio = (ratio_w, ratio_h)

        lines = []
        with open(text_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(' ')
                xs = list(map(float, line[0::2]))
                ys = list(map(float, line[1::2]))
                line = np.stack([xs, ys], axis=-1)
                lines.append(line)

        aligned_lines = []
        ts_list = []

        for line in lines:
            num_points, _ = line.shape
            index = np.random.randint(0, num_points, self.num)
            aligned_lines.append((line[index, :] * ratio).astype(np.float32))
            ts = index / (num_points - 1)
            ts_list.append(ts)

        existence = np.zeros(self.max_lane, dtype=np.float32)
        existence[:len(lines)] = 1.0
        padding = self.max_lane - len(lines)

        if padding == self.max_lane:
            xs = np.random.randint(0, self.input_size[0], self.num * self.max_lane)
            ys = np.random.randint(0, self.input_size[1], self.num * self.max_lane)
            aligned_lines = np.stack([xs, ys], axis=-1).reshape((self.max_lane, self.num, 2))
            aligned_lines = aligned_lines.astype(np.float32)
            ts_list = np.random.uniform(0.0, 1.0, self.max_lane * self.num).reshape((self.max_lane, self.num))
        else:
            for i in range(padding):
                aligned_lines.append(aligned_lines[-1])
                ts_list.append(ts_list[-1])
            aligned_lines = np.array(aligned_lines)
            ts_list = np.array(ts_list)

        return image, existence, aligned_lines, ts_list

    def __len__(self):
        return len(self.image_files)

    def get_num_fc_nodes(self):
        return (self.degree + 1) * 2 * self.max_lane


def build_dataloader(root, batch_size, input_size, degree, num, max_lane, use_gpu, num_workers):
    trainset = CULane(root, input_size, degree, num=num, max_lane=max_lane, split="train")
    valset = CULane(root, input_size, degree, num=num, max_lane=max_lane, split="val")
    testset = CULane(root, input_size, degree, num=num, max_lane=max_lane, split="test")

    tr_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu, drop_last=True,
                           num_workers=num_workers)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=use_gpu, drop_last=False,
                            num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=use_gpu, drop_last=False,
                             num_workers=num_workers)

    return tr_loader, val_loader, test_loader, trainset.get_num_fc_nodes()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CULane dataset lodaer.")
    parser.add_argument("--root", type=str, default="../CULane", help="Path to the root directory of the dataset.")
    parser.add_argument("--degree", type=int, default=5, help="Degree of the bezier curves.")
    parser.add_argument("--max_lane", type=int, default=4, help="Maximum number of lanes.")
    parser.add_argument("--num", type=int, default=72, help="The number of ts for each lane.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train", help="Dataset split.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--input_size", type=int, nargs=2, default=(820, 295), help="Input image size (width, height).")

    args = parser.parse_args()
    print(args)

    tr_loader, val_loader, test_loader, num_fc_nodes = build_dataloader(args.root, args.batch_size, args.input_size,
                                                                        args.degree, args.num, args.max_lane, True, 12)

    for images, existence, aligned_lines, ts in tr_loader:
        pass
