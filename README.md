[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)

# IDET: Iterative Difference-Enhanced Transformers for High-Quality ChangeDetection
This repo presents [IDET](https://arxiv.org/pdf/2207.09240), an iterative Difference-Enhanced Transformer block for change detection. 
It consists of three Transformers, where first two transformers are used to enhance the reference and query features and the last to strength the difference features between them. 
The core of the model, the *terative Difference-Enhanced Transformers* (IDET) block, 
makes sure that the network can generate the high-quality difference features.

![IDET framework](img/framework.png#pic_center)

![IDET model architecture](img/IDET.png#pic_center)



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
@misc{https://doi.org/10.48550/arxiv.2207.09240,
  doi = {10.48550/ARXIV.2207.09240},
  url = {https://arxiv.org/abs/2207.09240},
  author = {Huang, Rui and Wang, Ruofei and Guo, Qing and Zhang, Yuxiang and Fan, Wei},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {IDET: Iterative Difference-Enhanced Transformers for High-Quality Change Detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```