[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)

# IDET: Iterative Difference-Enhanced Transformers for High-Quality ChangeDetection
This repo presents [IDET](https://arxiv.org/pdf/2207.09240), an iterative Difference-Enhanced Transformer block for change detection. 
It consists of three Transformers, where first two transformers are used to enhance the reference and query features and the last to strength the difference features between them. 
The core of the model, the *terative Difference-Enhanced Transformers* (IDET) block, 
makes sure that the network can generate the high-quality difference features.

![IDET framework](docs/model_all_simplified_cropped.svg)

![IDET model architecture](docs/model_all_simplified_cropped.svg)



### Setup

Python 3 dependencies:

* Pytorch 1.4.0
* torchvision 0.5.0
* numpy
* matplotlib



We .



---
## CD Datasets

### Street views datasets
#### VL-CMU-CD & PCD & CDnet2014


### Remote sensing datasets
#### LEVIR-CD & CDD & AICD

### Labels

---
## Cite


```
@inproceedings{andermatt2020weakly,
  title={A Weakly Supervised Convolutional Network for Change Segmentation and Classification},
  author={Andermatt, Philipp and Timofte, Radu},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```