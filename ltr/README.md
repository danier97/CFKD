# LTR

A general PyTorch based framework for learning tracking representations. 
## Table of Contents

* [Quick Start](#quick-start)
* [Overview](#overview)
* [Train Settings](#train-settings)
* [Training networks](#training-networks)

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate pytracking
python run_training.py train_module train_name
```
Here, ```train_module``` is the sub-module inside ```train_settings``` and ```train_name``` is the name of the train setting file to be used.

For example, you can train the student ATOM with MobileNetV2 using our CFKD method by running:
```bash
python run_training cfkd atom_cfkd
```
**Note**: Before training, the user needs to download the datasets and specify their paths in [admin/local.py]() that will be generated after setup. User also needs to specify the absolute path of the teacher in the training setting [files](train_settings).


## Overview
The framework consists of the following sub-modules.  
 - [actors](actors): Contains the actor classes for different trainings. The actor class is responsible for passing the input data through the network can calculating losses.  
 - [admin](admin): Includes functions for loading networks, tensorboard etc. and also contains environment settings.  
 - [dataset](dataset): Contains integration of a number of training datasets. Additionally, it includes modules to generate synthetic videos from image datasets. 
 - [data_specs](data_specs): Information about train/val splits of different datasets.   
 - [data](data): Contains functions for processing data, e.g. loading images, data augmentations, sampling frames from videos.  
 - [external](external): External libraries needed for training. Added as submodules.  
 - [models](models): Contains different layers and network definitions, as well as our developed [loss functions](models/loss/distillation.py).  
 - [trainers](trainers): The main class which runs the training.  
 - [train_settings](train_settings): Contains settings files, specifying the training of a network.   
 
## Train settings
 The training setting files include

- [bbreg](train_settings/bbreg) (atom_paper): The settings used in the original ATOM paper.
- [ablation](train_settings/ablation) (gt, ts, ts_ah, sep): The settings used for our ablation study of loss functions.
- [architecture](train_settings/architecture) (resnet18medium, resnet18small, resnet18tiny): The settings used for our study of backbone architectures.
- [cfkd](train_settings/architecture) (atom_cfkd, drnet_cfkd): The settings used for training with our CFKD method the student ATOM and DRNet trackers with MobileNetV2 backbones.
- [compression](train_settings/compression) (atom_compression): The settings used for training with [Wang et al](https://ieeexplore.ieee.org/abstract/document/9080535?casa_token=iOe3fxsyvN4AAAAA:BuEYvFcYvTIgBVWtqkKlokDs_D1WiGYnuMQaQXKo7aT2cV9kYyhaGNbTtJKIuBWwgL0Y5CgBpA)'s method the student ATOM tracker with MobileNetV2 backbone.
- [tskd](train_settings/tskd) (atom_tskd): The settings used for training with [Liu et al](https://arxiv.org/abs/1907.10586)'s method the student ATOM tracker with MobileNetV2 backbone.

## Training networks
To train a network using the toolkit, the following components need to be specified in the train settings. For reference, see [atom_cfkd.py](train_settings/cfkd/atom_cfkd.py).  
- Datasets: The datasets to be used for training. A number of standard tracking datasets are already available in ```dataset``` module.  
- Processing: This function should perform the necessary post-processing of the data, e.g. cropping of target region, data augmentations etc.  
- Sampler: Determines how the frames are sampled from a video sequence to form the batches.  
- Network: The network module to be trained.  
- Objective: The training objective.  
- Actor: The trainer passes the training batch to the actor who is responsible for passing the data through the network correctly, and calculating the training loss.  
- Optimizer: Optimizer to be used, e.g. Adam.  
- Trainer: The main class which runs the epochs and saves checkpoints. 
 

 