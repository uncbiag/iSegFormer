# iSegFormer: Interactive Image Segmentation via Transformers with Application to 3D Knee MR Images
PyTorch implementation for paper [iSegFormer: Interactive Image Segmentation via Transformers with Application to 3D Knee MR Images](https://arxiv.org/abs/2112.11325) (MICCAI 2022) <br>
Qin Liu,
Zhenlin Xu,
Yining Jiao,
Marc Niethammer <br>
UNC-Chapel Hill

<p align="center">
    <a href="https://arxiv.org/abs/2112.11325">
        <img src="https://img.shields.io/badge/arXiv-2102.06583-b31b1b"/>
    </a>
    <a href="https://colab.research.google.com/github/qinliuliuqin/iSegFormer/blob/main/notebooks/colab_test_isegformer.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="The MIT License"/>
    </a>
</p>

<p align="center">
  <img src="./assets/img/iSegFormer.png" alt="drawing", width="650"/>
</p>

## Update
`🔥[2024-Mar-8]:` We update the code of the following work on interactive volume segmentation in branch [v2.0](https://github.com/uncbiag/iSegFormer/tree/v2.0). 
> [Exploring Cycle Consistency Learning in Interactive Volume Segmentation](https://arxiv.org/abs/2303.06493). <br>
> Qin Liu<sup>1</sup>,
> Meng Zheng<sup>2</sup>,
> Benjamin Planche<sup>2</sup>,
> Zhongpai Gao<sup>2</sup>,
> Terrence Chen<sup>2</sup>,
> Marc Niethammer<sup>1</sup>, 
> Ziyan Wu<sup>2</sup> <br>
> <sup>1</sup>UNC-Chapel Hill, <sup>2</sup>United Imaging Intelligence</sup> <br>


## Installation
If you want to test our models remotely, run this [colab notebook](https://colab.research.google.com/github/qinliuliuqin/iSegFormer/blob/main/notebooks/colab_test_isegformer.ipynb
). Otherwise, you have to download our codebase and install it locally.

This framework is built using Python 3.9 and relies on the PyTorch 1.4.0+. The following command installs all 
necessary packages:

```.bash
pip3 install -r requirements.txt
```
If you want to run training or testing, you must configure the paths to the datasets in [config.yml](config.yml).

## Demo with GUI
```
$ ./run_demo.sh
```

## Evaluation
First, download the [datasets and pretrained weights](https://drive.google.com/drive/folders/1KG6QVwlydyEzmcKNHgvCZjKhsaycIFiV?usp=sharing) and run the following code for evaluation:
```
python scripts/evaluate_model.py NoBRS \
--gpu 0 \
--checkpoint=./weights/imagenet21k_pretrain_oaizib_finetune_swin_base_epoch_54.pth \
--datasets=OAIZIB
```

## Training
Train the Swin-B model on the OAIZIB dataset.
```
python train.py models/iter_mask/swinformer_large_oaizib_itermask.py \
--batch-size=22 \
--gpu=0
```

## Download: models and datasets
[Google Drive](https://drive.google.com/drive/folders/1KG6QVwlydyEzmcKNHgvCZjKhsaycIFiV?usp=sharing)

<!-- ## Datasets
[OAI-ZIB-test](https://github.com/qinliuliuqin/iSegFormer/releases/download/v0.1/OAI-ZIB-test.zip) \
[BraTS20](https://drive.google.com/drive/folders/12iSwrI2M98pV7s_5hOrp9r-PELlQzWOq?usp=sharing) \
[ssTEM](https://github.com/unidesigner/groundtruth-drosophila-vnc/tree/master/stack1/raw)
 -->
<!-- ## Video Demos
The following two demos are out of date.
[Demo 1: OAI Knee](https://drive.google.com/file/d/1HyQsWYA6aG7I5C57b8ZTczNrW9OR6ZDS/view?usp=sharing) \
[Demo 2: ssTEM](https://drive.google.com/file/d/1dZL91P2rDEQqrlHQi2XaTlnY1rmWezNF/view?usp=sharing)
 -->
 
## License
The code is released under the MIT License. It is a short, permissive software license. Basically, you can do whatever you want as long as you include the original copyright and license notice in any copy of the software/source. 

## Citation
```bibtex
@inproceedings{liu2022isegformer,
  title={iSegFormer: interactive segmentation via transformers with application to 3D knee MR images},
  author={Liu, Qin and Xu, Zhenlin and Jiao, Yining and Niethammer, Marc},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={464--474},
  year={2022},
  organization={Springer}
}
```
