# Contents

- [CPCNN-Net Description](#cpcnn-net-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Code Description](#code-description)
    - [File List](#file-list)
    - [Code Parameters](#code-parameters)
        - [Parameters Configuration](#parameters-configuration)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Training Result](#training-result)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
        - [Evaluation Result](#evaluation-result)
- [Description of Random Situation](#description-of-random-situation)

# [CPCNN-Net Description](#contents)

Paper: [Cross-Part Learning for Fine-Grained Image Classification ](https://ieeexplore.ieee.org/document/9656684)
Authors: Man Liu, Chunjie Zhang, Huihui Bai, Riquan Zhang, Yao Zhao.
Published in: IEEE Transactions on Image Processing ( Volume: 31) Page(s): 748 - 758 
Date of Publication: 20 December 2021 
DOI: 10.1109/TIP.2021.3135477

# [Model Architecture](#contents)

In order to improve the performance of fine-grained image classification (FGIC) without boundary box or dense component labeling, this paper proposes a weakly supervised cross-component convolution neural network model (CP-CNN). Firstly, the proposed feature enhancement module (FEB) improves the positioning ability of the part proposal generator  (PPG); After locating the parts, realize the joint feature learning between components through the context transformer; Finally, Pyramid Context Integration Block (PCIB) is used to integrate the fused features.

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [Caltech-UCSD Birds-200-2011](<http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>)

Please download the datasets [CUB_200_2011.tgz] and unzip it, then put all training images into a directory named "train", put all testing images into a directory named "test".

The directory structure is as follows:

```path
.
└─cub_200_2011
  ├─train
  └─test
```

# [Environment Requirements](#contents)

- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Code Description](#contents)
Many thanks for [NTS-NET](https://gitee.com/hit_zyh/ntsnet).A part of the code is borrowed from it.
## [File List](#contents)

```shell
.
└─ntsnet
  ├─README.md                             # README
  ├─src
    ├─config.py                           # network configuration
    ├─dataset.py                          # dataset utils
    ├─lr_generator.py                     # leanring rate generator
    ├─network.py                          # network define for ntsnet
    └─my_resnet.py                        # resnet.py
  ├─eval.py                               # evaluation scripts
  └─train.py                              # training scripts
```

## [Code Parameters](#contents)

### [Parameters Configuration](#contents)

```txt
"img_width": 448,           # width of the input images
"img_height": 448,          # height of the input images


# LR
"base_lr": 0.001,                                                              # base learning rate
"num_epochs": 200,                                                             # total epoch in lr generator
"momentum": 0.9,                                                               # sgd momentum in optimizer

# train
"topK": 6,                                                                     # topK parts
"batch_size": 8,
"num_classes": 200，
"learning_rate": 0.001,
"weight_decay": 1e-4,
"epoch_size": 200,                                                             # total epoch size

# checkpoint
"save_checkpoint": True,                                                       # whether save checkpoint or not
"save_checkpoint_epochs": 5,                                                   # save checkpoint interval
"keep_checkpoint_max": 10,
"num_train_images": 5994,                                                      # train images number in dataset
"num_test_images": 5794,                                                       # test images number in dataset

```

## [Training Process](#contents)

- Set options in `config.py`, including learning rate, output filename and network hyperparameters. 

### [Training](#contents)

- Run `train.py` for training of CPCNN-Net model.

- Notes
1. As for PRETRAINED_MODEL，it should be a trained ResNet50 checkpoint.

### [Training Result](#contents)

Training result will be stored in train_url path. You can find checkpoint file together with result. 
The directory structure is as follows:

```shell
.
└─cache
  ├─ckpt_0
    ├─cpcnnnet-101_100.ckpt
  ├─summary_dir
    ├─events.out.events.summary.1676516343.0.hostname_lineage
    ├─events.out.events.summary.1676516343.0.hostname_MS
```

After training, start MindInsight to visualize the collected data.For more information, please check the resources [MindSpore summary](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0.0-alpha/summary_record.html?highlight=summary_collector)



## [Evaluation Process](#contents)

### [Evaluation](#contents)

- Run `eval.py` for evaluation.

### [Evaluation Result](#contents)

Inference result will be stored in the train_url path. Under this, you can find result like the following in eval.log.

```bash
ckpt file name: cpcnnnet-180_749.ckpt
accuracy:  0.916 
```
We provide the model parameters trained on CUB_200_2011. You can download it [here](https://pan.baidu.com/s/1NQdoOq8FBpNHFqujRnfQJw?pwd=jpfa)

# [Description of Random Situation](#contents)

We use random seed in train.py and eval.py for weight initialization.
