[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)

# IDET: Iterative Difference-Enhanced Transformers for High-Quality Change Detection
This repo presents [IDET](https://arxiv.org/pdf/2207.09240), an iterative Difference-Enhanced Transformer block for change detection. 
It consists of three Transformers, where the first two transformers are used to enhance the reference and query features and the last to strengthen the difference features between them. 
The core of the model, the *iterative Difference-Enhanced Transformers* (IDET) block, 
makes sure that the network can generate high-quality difference features.

<table frame=void>	
	<tr>		   <!--<tr>一行的内容<\tr>，<td>一个格子的内容<\td>-->
    <td><center><img src="img/framework.png"		
                     alt="IDET framework"
                     height="400px"/>
        <br>"IDET Framework."
        </center></td>	
    <td><center><img src="img/IDET.png"
                     alt="IDET architecture"
                     height="400px"/><br>"IDET Architecture."</center></td>
    </tr>
</table>




### Setup

Python 3 dependencies:

* Pytorch 1.4.0

* torchvision 0.5.0

* numpy

* matplotlib

  

---
## CD Datasets

We conduct various experiments on six widely used change detection datasets, which can be divided into two types, street views CD datasets and remote sensing scenes CD datasets. Our data structure as follows:

```
|-- data_CMU 
|----|- train            # training dataset
|        |- image.txt    # reference image path
|        |- image2.txt   # query image path
|        |- label.txt    # ground truth path
|----|-test              # testing dataset
|        |- image.txt
|        |- image2.txt
|        |- label.txt
.
```

## Street views datasets:
VL-CMU-CD: A street-view change detection dataset.
PCD:  Tsunami destroyed views (Tsunami) and Google Street Views (GSV). 
CDnet2014: Frame-based change detection benchmark dataset.

## Remote sensing datasets:
LEVIR-CD: Buildings change detection dataset.
CDD:  Complex scenes aerial and satellite imagery change detection dataset. 
AICD: Aerial image change detection dataset.

## Labels

We all use the binary change maps to supervise the training of IDET. 

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