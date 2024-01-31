# SARFormer: Large-Vision Model-Guided Transformer for Semantic Segmentation




## Abstract
Dense prediction tasks, such as semantic segmentation, grouping pixels into different classes. Recently, research has shifted from per-pixel prediction to cluster prediction with the emergence of transformers. Mask Transformers are noteworthy as they predict mask labels directly. Furthermore, the newly presented Large Language Models (LVMs) enabling effective segmentation in all images. However, current research is primarily focused on designing specialized architectures for fixed datasets and distributions. This results in models that achieve high scores but show suboptimal performance in practical applications. Furthermore, LVM lacks good control over instance granularity, and overly detailed masks prevent its direct use in semantic segmentation. These limitations can have knock-on consequences in safety-critical systems. To address this issue, we present SARFormer, a new architecture capable of addressing semantic segmentation tasks on any dataset with less training. SARFormer is designed to further optimize SAM’s ability. SARFormer was evaluated on public benchmarks ADE20k and Cityscapes using various training settings. The results confirm its effectiveness.


![](resources\topbar.jpg)

## Installation

- See [MMsegmentation installation instructions](docs\en\get_started.md) to install mmseg. 

- Build ms_deform_attn: 

  ```
  cd ops
  bash make.sh
  ```

- Install Segment Anything

  ```
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

- Install requirements

  ```
  pip install -r requirement.txt
  ```

- Download SAM pre-train weights, and place at ./checkpoints

  - **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
  - `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
  - `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

  

## Getting Started

See [Preparing Datasets for MMseg](docs\en\user_guides\2_dataset_prepare.md).



## Model Zoo and Baselines

We provide the baseline results and trained models available for download.



## Pretraining Sources for Baseline

| Name | Year | Type | Data         | Repo                                                        | Paper                                     |
| ---- | ---- | ---- | ------------ | ----------------------------------------------------------- | ----------------------------------------- |
| BEiT | 2021 | MIM  | ImageNet-22K | [repo](https://github.com/microsoft/unilm/tree/master/beit) | [paper](https://arxiv.org/abs/2106.08254) |



## ADE20K 

### 

|   Method    |   Backbone    | Lr schd | Crop Size |                             mIoU                             |                            Config                            |                           Download                           |
| :---------: | :-----------: | :-----: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Mask2Former | ViT-Adapter-B |  160k   |    640    | [56.51](https://pan.baidu.com/s/1Kd__Mi1HXrzg2k-8I4gPtg?pwd=tb22) | [config](./configs/mask2former/mask2former_beit_adapter_large_640_160k_bs4_ade20k) | [ckpt](https://pan.baidu.com/s/1OSYdrbzq9TB9Inn1YK_u9A?pwd=q7gq) \| [log](https://pan.baidu.com/s/13-ipfAlLXW90VifLc9WQKQ?pwd=tqnn) \|  [weight](https://pan.baidu.com/s/1EkJGAMxARpQZf8mbS5i6CA?pwd=60sc) |
|  SARFormer  | ViT-Adapter-B |  160k   |    640    | [56.82](https://pan.baidu.com/s/1gtBYBxCg2WRFCBWiRSs9jQ?pwd=s4u1) | [config](configs\sarformer\sarformer_beit_base_640_160k_bs4_ade20k.py) | [ckpt](https://pan.baidu.com/s/1jsS3Wa0eLQgiXCKEnitrng?pwd=2fj4) \| [log](https://pan.baidu.com/s/1WZBYlZ2aKcHpK1Nutd1t4g?pwd=6iu3) \| [weight](https://pan.baidu.com/s/1EkJGAMxARpQZf8mbS5i6CA?pwd=60sc) |
|  SARFormer  | ViT-Adapter-L |  160k   |    640    | [57.21](https://pan.baidu.com/s/1U2JMGF3nKcS0-QLzMbdDbQ?pwd=11po) | [config](configs\sarformer\sarformer_beit_large_640_160k_bs4_ade20k.py) | [ckpt](https://pan.baidu.com/s/1uvF6NWfyPh82yMNCfhX0DA?pwd=dkai) \| [log](https://pan.baidu.com/s/1U2nLBLtB6Vt9wIR0g1ua4Q?pwd=knlz) \| [weight](https://pan.baidu.com/s/17_Lsb4ujXAMSq2Rq9JNcEg?pwd=kii6) |

