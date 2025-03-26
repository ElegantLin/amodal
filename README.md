# Open-World Amodal Appearance Completion (CVPR2025)

<p align="center">
  <p align="center" margin-bottom="0px">
    <a href="https://jiayangao.github.io/"><strong>Jiayang Ao</strong></a>
    ·
    <a href="https://cis.unimelb.edu.au/research/artificial-intelligence/ai-students/artificial-intelligence/yanbei-jiang"><strong>Yanbei Jiang</strong></a>
    ·
    <a href="https://research.monash.edu/en/persons/qiuhong-ke/"><strong>Qiuhong Ke</strong></a>
    ·
    <a href="http://www.kehinger.com/"><strong>Krista A. Ehinger</strong></a>
    <p align="center">
    <a href="https://arxiv.org/abs/2411.13019" style="text-decoration:none;">
      <img src="https://img.shields.io/badge/arXiv-2411.13019-b31b1b.svg" alt="arXiv Badge">
    </a>
    <a href="https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers" style="text-decoration:none;">
      <img src="https://img.shields.io/badge/Pub-CVPR'25-blue" alt="CVPR Badge">
    </a>
    <a href="https://opensource.org/license/apache-2-0" style="text-decoration:none;">
      <img src="https://img.shields.io/badge/License-Apache 2.0-yellow.svg" alt="apache-2-0 License Badge">
    </a>
  </p>
</p>

## Introduction
<p align="justify">
We introduce a training-free framework that expands amodal completion capabilities by accepting flexible text queries as input. Our approach generalizes to arbitrary objects specified by both direct terms and abstract queries. We term this capability reasoning amodal completion, where the system <strong>reconstructs the full appearance of the queried object based on the provided image and language query</strong>. Our framework unifies segmentation, occlusion analysis, and inpainting to handle complex occlusions and generates completed objects as RGBA elements, enabling seamless integration into applications such as 3D reconstruction and image editing.
</p>


![](figure/intro.png)

## Methodology
<p align="justify">
Overview of our framework. Starting with a text query, a VLM generates a visible mask to locate the target object in the input image. The framework then identifies all objects and background segments for occlusion analysis. An auto-generated prompt guides the inpainting model, which iteratively reconstructs the occluded object to produce a transparent RGBA amodal completion output.
</p>

![](figure/method.png)

## Requirements
Python: 3.10.14

PyTorch: 1.13.1+cu117

### Download and set up the pre-trained models:

1. ***LISA*** (for mapping natural language queries to visible object regions): Install LISA following the instructions from the [official LISA repository](https://github.com/dvlab-research/LISA). Then, download the checkpoint [LISA-13B-llama2-v1](https://huggingface.co/xinlai/LISA-13B-llama2-v1) from Hugging Face and place it under:`LISA/LISA-13B-llama2-v1/`. Note: We access LISA via API to avoid dependency conflicts. You can modify the LISA server URL in `main.py` by changing the LISA_SERVER_URL variable.

**⚠️ Replace the original LISA/app.py**  with [our modified version](https://github.com/saraao/amodal/blob/main/LISA/app.py) in this repository. This modified version introduces minimal changes to [line 310](https://github.com/saraao/amodal/blob/main/LISA/app.py#L310) and [line 322](https://github.com/saraao/amodal/blob/main/LISA/app.py#L322) to return the raw segmentation mask (pred_mask) for integration with our pipeline.

2. ***InstaOrder*** (for occlusion relationships): Install InstaOrder following the instructions from the [InstaOrder repository](https://github.com/POSTECH-CVLab/InstaOrder), download the checkpoint `InstaOrder_InstaOrderNet_od` and place it under:`InstaOrder/InstaOrder_ckpt/`.

3. ***RAM-Grounded-SAM***: Install RAM++ following the instructions from the [official recognize-anything repository](https://github.com/xinyu1205/recognize-anything), download the checkpoint `ram_plus_swin_large_14m`. Install Grounded-SAM following the instructions from the [official Grounded-Segment-Anything repository](https://github.com/IDEA-Research/Grounded-Segment-Anything), download the checkpoint `groundingdino_swint_ogc`.

4. ***Stable Diffusion***: [Stable Diffusion v2 inpainting model](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting).

## Usage

This repository contains the implementation of our pipeline. Please refer to `main.py` for the core implementation. We will update the documentation and remainings soon.

Stay tuned for updates!

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](https://github.com/saraao/amodal/blob/main/LICENSE) file for details.

### Acknowledgments

We thank the following papers for their open-source code, pre-trained models and datasets:
- Amodal Completion via Progressive Mixed Context Diffusion [[CVPR 2024]](https://github.com/k8xu/amodal)
- LISA: Reasoning Segmentation via Large Language Model [[CVPR 2024]](https://github.com/dvlab-research/LISA)  
- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection [[ECCV 2024]](https://github.com/IDEA-Research/GroundingDINO)
- Segment anything [[CVPR 2023]](https://github.com/facebookresearch/segment-anything)
- Open-set image tagging with multi-grained text supervision [[arXiv 2023]](https://github.com/xinyu1205/recognize-anything)
- Instance-wise occlusion and depth orders in natural scenes [[CVPR 2022]](https://github.com/POSTECH-CVLab/InstaOrder)
- High-resolution image synthesis with latent diffusion models [[CVPR 2022]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
- Learning transferable visual models from natural language supervision [[PMLR 2021]](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)
- Semantic amodal segmentation [[CVPR 2017]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Semantic_Amodal_Segmentation_CVPR_2017_paper.pdf)
- Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations [[IJCV 2017]](https://dl.acm.org/doi/10.1007/s11263-016-0981-7)
- LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs [[Data Centric AI NeurIPS Workshop 2021]](https://laion.ai/blog/laion-400-open-dataset/)

## Citation

If you find this helpful in your work, please consider citing our paper:
```
@article{ao2024open,
  title={Open-World Amodal Appearance Completion},
  author={Ao, Jiayang and Jiang, Yanbei and Ke, Qiuhong and Ehinger, Krista A},
  journal={arXiv preprint arXiv:2411.13019},
  year={2024}
}
```

# Contact
If you have any questions regarding this work, please send email to jiayang.ao@student.unimelb.edu.au.
