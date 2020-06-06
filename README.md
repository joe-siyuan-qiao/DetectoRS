# DetectoRS – MMDetection V2.0

## Introduction

This repo holds the code for [DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution](https://arxiv.org/pdf/2006.02334.pdf).
The project is based on [mmdetection-v2](https://github.com/open-mmlab/mmdetection).
Please refer to [mmdetection readme](README.mmdet.md) for installation and running scripts.
Please see the master branch for the original implementation based on mmdetection-v1.

## Results on COCO test-dev

| Method    | Detector          | lr | Config | Log | Model | Box AP | Mask AP |
|-----------|:-----------------:|:--:|:--------------:|:--------------:|:--------------:|:------------:|:------------:|
| RFP  | Cascade + ResNet-50 | 1x | [File Link](configs/detectors/cascade_rcnn_r50_rfp_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/cascade_rcnn_r50_rfp_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/cascade_rcnn_r50_rfp_1x_coco-8713b310.pth) | 45.0 | – |
| SAC  | Cascade + ResNet-50 | 1x |  [File Link](configs/detectors/cascade_rcnn_r50_sac_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/cascade_rcnn_r50_sac_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/cascade_rcnn_r50_sac_1x_coco-a6e88a40.pth) | 45.0 | – |
| DetectoRS  | Cascade + ResNet-50 | 1x |  [File Link](configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/detectors_cascade_rcnn_r50_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/detectors_cascade_rcnn_r50_1x_coco-7b6ec977.pth) | 47.5 | – |
| RFP  | HTC + ResNet-50 | 1x |  [File Link](configs/detectors/htc_r50_rfp_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/htc_r50_rfp_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/htc_r50_rfp_1x_coco-4357af3e.pth) | 46.3 | 40.5 |
| SAC  | HTC + ResNet-50 | 1x |  [File Link](configs/detectors/htc_r50_sac_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/htc_r50_sac_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/htc_r50_sac_1x_coco-cfbac01d.pth) | 46.5 | 41.0 |
| DetectoRS  | HTC + ResNet-50 | 1x |  [File Link](configs/detectors/detectors_htc_r50_1x_coco.py) | [Log](https://cs.jhu.edu/~syqiao/DetectoRS/detectors_htc_r50_1x_coco.json) | [Model](https://cs.jhu.edu/~syqiao/DetectoRS/detectors_htc_r50_1x_coco-ac1ebf3a.pth) | 49.0 | 42.7 |

## Citing DetectoRS

If you think DetectoRS is useful in your project, please consider citing us.

```BibTeX
@article{detectors,
  title={DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  journal={arXiv preprint arXiv:2006.02334},
  year={2020}
}
```
