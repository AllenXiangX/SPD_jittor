# Point Cloud Completion

This repository contains the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation for the papers:

1. Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2022)

2. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021, Oral)

[<img src="../pics/completion.png" width="100%" alt="Intro pic" />](../pics/completion.png)

## Datasets

We use the [PCN](https://www.shapenet.org/), [ShapeNet-34/21](https://github.com/yuxumin/PoinTr), and [Compeletion3D](http://completion3d.stanford.edu/) datasets in our experiments, which are available below:

- [PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [ShapeNet-34/21](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md)
- [Completion3D](https://completion3d.stanford.edu/)

## Getting Started

To use our code, make sure that the environment is installed according to the instructions in the [main page](https://github.com/AllenXiangX/SPD_jittor). Then modify the dataset path in the [configuration files](https://github.com/AllenXiangX/SPD_jittor/tree/main/completion/configs).


## Training

To train a point cloud completion model from scratch, run:

```
export CUDA_VISIBLE_DEVICES='0'
python train.py --configs <config>
```

For example:

```
export CUDA_VISIBLE_DEVICES='0'
python train.py --configs ./configs/pcn_cd1.yaml
```

## Evaluation

To evaluate a pre-trained model, first specify the model_path in configuration file, then run:

```
export CUDA_VISIBLE_DEVICES='0'
python test.py --configs <config>
```

For example:

```
export CUDA_VISIBLE_DEVICES='0'
python test.py --configs ./configs/pcn_cd1.yaml
```

## Acknowledgements


This repo is based on: 
- [GRNet](https://github.com/hzxie/GRNet), 
- [PoinTr](https://github.com/yuxumin/PoinTr),

We thank the authors for their great job!