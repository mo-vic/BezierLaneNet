'''
This script reads labels from .txt files
and paint it on the corresponding images.
@author: movic
@date: 2020-07-17
'''

import os
import glob
import argparse

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Draw lanes on the images.")
    parser.add_argument("--root", type=str, default="../CULane", help="Path to the dataset.")
    parser.add_argument("--count", type=int, default=20, help="Number of image to visualize.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.root)
    assert args.count > 0
    image_files = glob.glob(os.path.join(args.root, "**/*.jpg"), recursive=True)
    assert len(image_files)

    np.random.seed(args.seed)
    r = np.random.permutation(len(image_files))
    image_files = np.array(image_files)
    image_files = image_files[r]
    image_files = image_files.tolist()

    for image_file in image_files[:args.count]:
        image = cv2.imread(image_file)
        text_file = image_file.replace(".jpg", ".lines.txt")
        assert os.path.exists(text_file)
        lines = []
        with open(text_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(' ')
                xs = list(map(float, line[0::2]))
                ys = list(map(float, line[1::2]))
                line = np.stack([xs, ys], axis=-1)
                line = line.astype(np.int32)
                lines.append(line.tolist())
        for line in lines:
            for center in line:
                center = tuple(center)
                cv2.circle(image, center, radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow("lane", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
