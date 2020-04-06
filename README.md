# DetectoRS

## Introduction

This repo holds the code for [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/abs/2006.????).
The project is based on [mmdetection codebase](https://github.com/open-mmlab/mmdetection).
Please refer to [mmdetection readme](README.mmdet.md) for installation and running scripts.

## Main Results on COCO test-dev

| Method    | Backbone          | Config | Model | Box AP | Mask AP |
|-----------|:-----------------:|--------------|--------------|:------------:|:------------:|
| DetectoRS | ResNet-50         | [File Link](configs/DetectoRS/DetectoRS_mstrain_400_1200_r50_40e.py) | [Download](http://cs.jhu.edu/~syqiao/DetectoRS/DetectoRS_R50-0f1c8080.pth) | 51.3 | 44.4 |
| DetectoRS | ResNeXt-101-32x4d | [File Link](configs/DetectoRS/DetectoRS_mstrain_400_1200_x101_32x4d_40e.py) | [Download](https://www.cs.jhu.edu/~syqiao/DetectoRS/DetectoRS_X101-ed983634.pth) | 53.3 | 45.8 |

## Citing DetectoRS

If you think DetectoRS is useful in your project, please consider citing us.

```BibTeX
@article{detectors,
  title={DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  journal={arXiv preprint arXiv:2006.????},
  year={2020}
}
```
