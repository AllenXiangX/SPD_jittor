# Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2023) (Jittor implementation)

[Peng Xiang*](https://scholar.google.com/citations?user=Bp-ceOAAAAAJ&hl=zh-CN&oi=sra), [Xin Wen*](https://scholar.google.com/citations?user=7gcGzs8AAAAJ&hl=zh-CN&oi=sra), [Yu-Shen Liu](http://cgcad.thss.tsinghua.edu.cn/liuyushen/), [Yan-Pei Cao](https://scholar.google.com/citations?user=50194vkAAAAJ&hl=en), Pengfei Wan, Wen Zheng, [Zhizhong Han](https://h312h.github.io/)

[<img src="./pics/SPD.png" width="100%" alt="Intro pic" />](pics/SPD.png)



## [SPD]

This repository contains the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of the papers:

**1. Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2023)**

**2. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021, Oral)**


The **Jittor** implementation of SPD on different tasks. Compared to the PyTorch implementation, the Jittor version has higher training speed (green denotes faster training speed under the same experimental settings):
| task        |  dataset | PyTorch | Jittor |
| ----------- | -------------- | ------------ | ------------ |
| Completion  | PCN | 10min 41s | 	$\textbf{\color{YellowGreen}9min 46s}$ |
| Completion  | Completion3D | 6min 0s  | $\textbf{\color{YellowGreen}4min 50s}$ |
| Completion  | ShapeNet-34/21 | 3min 35s | $\textbf{\color{YellowGreen}3min 29s}$  |
| AE          | ShapeNet | 1min 6s | $\textbf{\color{YellowGreen}56s}$  |
| VAE         | ShapeNet | 1min 9s | $\textbf{\color{YellowGreen}57s}$  |
| Up-sampling | PUGAN    | 2min 34s | $\textbf{\color{YellowGreen}2min 2s}$  |
| SVR | ShapeNet | 27s | $\textbf{\color{YellowGreen}23s}$  | 


[ [SnowflakeNet](https://openaccess.thecvf.com/content/ICCV2021/html/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.html) | [SPD](https://arxiv.org/abs/2202.09367) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9928787) | [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) ] 

> Most existing point cloud completion methods suffer from the discrete nature of point clouds and the unstructured prediction of points in local regions, which makes it difficult to reveal fine local geometric details. To resolve this issue, we propose SnowflakeNet
with snowflake point deconvolution (SPD) to generate complete point clouds. SPD models the generation of point clouds as the
snowflake-like growth of points, where child points are generated progressively by splitting their parent points after each SPD. Our
insight into the detailed geometry is to introduce a skip-transformer in the SPD to learn the point splitting patterns that can best fit the
local regions. The skip-transformer leverages attention mechanism to summarize the splitting patterns used in the previous SPD layer
to produce the splitting in the current layer. The locally compact and structured point clouds generated by SPD precisely reveal the
structural characteristics of the 3D shape in local patches, which enables us to predict highly detailed geometries. Moreover, since
SPD is a general operation that is not limited to completion, we explore its applications in other generative tasks, including point cloud auto-encoding, generation, single image reconstruction, and upsampling. Our experimental results outperform state-of-the-art methods under widely used benchmarks.

## [Citation]

```
@ARTICLE{xiang2022SPD,
  author={Xiang, Peng and Wen, Xin and Liu, Yu-Shen and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Han, Zhizhong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer}, 
  year={2022},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2022.3217161}}

@inproceedings{xiang2021snowflakenet,
  title={{SnowflakeNet}: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer},
  author={Xiang, Peng and Wen, Xin and Liu, Yu-Shen and Cao, Yan-Pei and Wan, Pengfei and Zheng, Wen and Han, Zhizhong},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## [Getting Started]

#### Build Environment

```bash
# python environment
$ cd SnowflakeNet
$ conda create -n spd_jittor python=3.7
$ conda activate spd_jittor
$ pip3 install -r requirements.txt

# Jittor
$ sudo apt install python3.7-dev libomp-dev
$ python3.7 -m pip install jittor
```


## Visualization of point splitting paths
We provide visualization code for point splitting paths in the *visualization* folder.


## Acknowledgements

Some of the code of this repo is borrowed from: 
- [PointCloudLib](https://github.com/Jittor/PointCloudLib)
- [GRNet](https://github.com/hzxie/GRNet)
- [PoinTr](https://github.com/yuxumin/PoinTr)
- [diffusion-point-cloud](https://github.com/luost26/diffusion-point-cloud),
- [3DAttriFlow](https://github.com/junshengzhou/3DAttriFlow)
- [PUGAN](https://github.com/liruihui/PU-GAN)
- [pytorchpointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)
- [EMD](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd)


We thank the authors for their great job!

## License

This project is open sourced under MIT license.
