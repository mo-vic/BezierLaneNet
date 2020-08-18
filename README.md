# BezierLaneNet
Exploring the use of bezier curves for lane detection —— A baseline model.

![](./images/visualization.png)



演示视频和博客地址：https://mo-vic.github.io/2021/02/21/BezierLaneNet/



## Dataset

Download **CULane** dataset [here](https://xingangpan.github.io/projects/CULane.html).

## Installation

1. install [PyTorch](https://pytorch.org/);
2. run the following command:

```shell
pip3 install -r requirements.txt
```

## Training

Command-line arguments to reproduce the result:

```shell
python3 train.py --data ./CULane --log_name baseline --pretrained_weight ./weights/resnet18-5c106cde.pth --input_size 820 295 --gpu_ids 0
```

**Note**: make sure that you've downloaded the weight for ResNet18 pre-trained on [ImageNet](http://image-net.org/).

## Inference

Run the following command:

```shell
python3 inference.py --data ./CULane --ckpt [Path to the model checkpoint file] --input_size 820 295 --gpu_ids 0 --save_name "./video/output.avi"
```

## Possible Directions for Improvement

#### A. Pay More Attention to the curved lanes:

1. Use Focal loss;
2. Adaptive weighting according to the curvature;
3. Data resampling;
4. ......

#### B. Apply Data Augmentation 

#### C. Try Using Different Loss Function

#### D. Incorporate with Temporal Information:

1. Training on Video Data;
2. Use Consistent Loss;
3. ......

#### E. ......



**And Let Me Know If You Have Any Progress in These Directions!!!**

