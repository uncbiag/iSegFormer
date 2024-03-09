# Exploring Cycle Consistency Learning in Interactive Volume Segmentation
PyTorch implementation for paper [Exploring Cycle Consistency Learning in Interactive Volume Segmentation](https://arxiv.org/abs/2303.06493).

Qin Liu<sup>1</sup>,
Meng Zheng<sup>2</sup>,
Benjamin Planche<sup>2</sup>,
Zhongpai Gao<sup>2</sup>,
Terrence Chen<sup>2</sup>,
Marc Niethammer<sup>1</sup>, 
Ziyan Wu<sup>2</sup>
<br>
<sup>1</sup>UNC-Chapel Hill, <sup>2</sup>United Imaging Intelligence</sup>

#### [Paper](https://arxiv.org/abs/2303.06493) | [Demo Videos](https://drive.google.com/drive/folders/1bPLn7ZsZB3xRKNhxOB0ewWX3rlxp2pK_?usp=sharing)

<p align="center">
  <img src="./assets/framework.png" alt="drawing", width="850"/>
</p>

## Installation
The code is tested with ``python=3.9``, ``torch=1.12.0``, and ``torchvision=0.13.0`` on an A6000 GPU.
```
git clone https://github.com/uncbiag/iSegFormer
cd iSegFormer
git checkout v2.0
conda create -n isegformer python=3.9
conda activate isegformer
pip3 install -r requirements.txt
```

## Getting Started
First, download model weights and a medical volume for demo purposes.
```
python download.py
```
Then run a demo:
```
./run_demo.sh
```
Below is an example

