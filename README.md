# XAIPalette-Exploring-Image-Structure-via-Color-Distillation-for-Explainable-AI

Author: Manai Mortadha 

## About
This repository contains the PyTorch implementation of XAIPalette, a novel approach aiming to explore image structure via Color Distillation for Explainable AI (XAI). It provides insights into image interpretation by distilling structural information through color representation.

## Overview
XAIPalette focuses on decoding image structures using Color Distillation, as detailed in our paper *[Learning to Structure an Image with Few Colors](https://hou-yz.github.io/publication/2019-cvpr2020-colorcnn)*, which introduces ColorCNN, an innovative architecture for image color quantization.
![System Overview](https://hou-yz.github.io/images/ColorCNN_system.png "System overview depicting image color quantization with ColorCNN")

## Content
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Code](#code)
    * [Training Classifiers](#training-classifiers)
    * [Training & Evaluating ColorCNN](#training--evaluating-colorcnn)
    * [Evaluating Traditional Methods](#evaluating-traditional-methods)

## Dependencies
The code requires the following libraries:
- Python 3.7+
- PyTorch 1.4+ & torchvision
- NumPy
- Matplotlib
- Pillow
- OpenCV-Python

## Data Preparation
The default datasets (CIFAR10, CIFAR100, STL10, and tiny-imagenet-200) are stored in `~/Data/`. 
For Tiny-imagenet-200, download it from this [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip). Extract the files under `~/Data/tiny200/`, then execute `python color_distillation/utils/tiny_imagenet_val_reformat.py` to reformat the validation set (credit to [@tjmoon0104](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/utils/tiny-imgnet-val-reformat.ipynb) for the code).

Your `~/Data/` folder should have this structure:

# Learning to Structure an Image with Few Colors [[Website](https://hou-yz.github.io/publication/2019-cvpr2020-colorcnn)] [[arXiv](https://arxiv.org/abs/2003.07848)]

```
@inproceedings{hou2020learning,
  title={Learning to Structure an Image with Few Colors},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10116--10125},
  year={2020}
}
```


Tiny-imagenet-200 can be downloaded from this [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip). 
Once downloaded, please extract the zip files under `~/Data/tiny200/`. 
Then, run `python color_distillation/utils/tiny_imagenet_val_reformat.py` to reformat the validation set. (thank [@tjmoon0104](https://github.com/tjmoon0104/Tiny-ImageNet-Classifier/blob/master/utils/tiny-imgnet-val-reformat.ipynb) for his code).

Your `~/Data/` folder should look like this
```
Data
├── cifar10/
│   └── ...
├── cifar100/ 
│   └── ...
├── stl10/
│   └── ...
└── tiny200/ 
    ├── train/
    │   └── ...
    ├── val/
    │   ├── n01443537/
    │   └── ...
    └── ...
```

## Code
One can find classifier training & evaluation for traditional color quantization methods in `grid_downsample.py`.
For ColorCNN training & evaluation, please find it in `color_cnn_downsample.py`. 

### Training Classifiers
In order to train classifiers, please specify `'--train'` in the arguments. 
```shell script
python grid_downsample.py -d cifar10 -a alexnet --train
``` 
One can run the shell script `bash train_classifiers.sh` to train AlexNet on all four datasets. 

### Training & Evaluating ColorCNN
Based on the original image pre-trained classifiers, we then train ColorCNN under specific color space sizes. 
```shell script
python color_cnn_downsample.py -d cifar10 -a alexnet --num_colors 2
``` 
Please run the shell script `bash train_test_colorcnn.sh` to train and evaluate *ColorCNN* with AlexNet on all four datasets, under a 1-bit color space. 

### Evaluating Traditional Methods
Based on pre-trained classifiers, one can directly evaluate the performance of tradition color quantization methods. 
```shell script
python python grid_downsample.py -d cifar10 -a alexnet --num_colors 2 --sample_type mcut --dither
``` 
Please run the shell script `bash test_mcut_dither.sh` to evaluate *MedianCut+Dithering* with AlexNet on all four datasets, under a 1-bit color space. 


## Inspiration Authors (source) :
- Hou, Yunzhong
- Zheng, Liang
- Gould, Stephen
- Manai, Mortadha



# XAIPalette-Exploring-Image-Structure-via-Color-Distillation-for-Explainable-AI
