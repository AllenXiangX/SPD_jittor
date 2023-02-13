# Single View Reconstruction

This repository contains the [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) implementation of single view reconstruction of the papers:

1. Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2022)

2. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021, Oral)

[<img src="../pics/svr.png" width="70%" alt="Intro pic" />](../pics/svr.png)

## Datasets

We follow [3DAttriFlow](https://github.com/junshengzhou/3DAttriFlow) and evaluate the single view reconstruction quality of SPD on the ShapeNet dataset.

- [ShapeNet](https://drive.google.com/drive/folders/1If_-t0Aw9Zps-gj5ttgaMSTqRwYms9Ag?usp=sharing)


## Getting Started

To use our code, make sure that the environment is installed according to the instructions in the [main page](https://github.com/AllenXiangX/SPD_jittor). Then modify the dataset path in the [configuration files](https://github.com/AllenXiangX/SPD_jittor//tree/main/svr/cfgs).


## Training

To train a svr model from scratch, run:

```
export CUDA_VISIBLE_DEVICES='0'
python train_svr.py -c cfgs/SVR.yaml
```


## Evaluation

To evaluate a pre-trained model, first specify the "load_model" path in cfgs/SVR.yaml, then run:

```
export CUDA_VISIBLE_DEVICES='0'
python val_svr.py -c cfgs/SVR.yaml
```

## Acknowledgements

This repo is based on: 
- [3DAttriFlow](https://github.com/junshengzhou/3DAttriFlow)

We thank the authors for their great job!