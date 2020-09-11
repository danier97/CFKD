# CFKD: Correlation Filter Knowledge Distillation

## Introduction
A knowledge distillation method named CFKD developed for [ATOM](https://openaccess.thecvf.com/content_CVPR_2019/html/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.html)-based trackers.


## Installation

### Clone the GIT repository.  
```bash
git clone https://github.com/danielism97/CFKD.git
```
   
### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```  
### Install dependencies and set up environment
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```  
This script will also download the default networks and set-up the environment. Now the user needs to specify the paths in the configuration files generated (see [here](pytracking/README.md) and [here](ltr/README.md))


## Evaluation and Training

#### [pytracking](pytracking) - for evaluating trackers

#### [ltr](ltr) - for training trackers


## Models and Results
Our models and results can be found from the shared [folder](https://drive.google.com/drive/folders/19mDhiPbQlxCUtnB-qy536RMQ6mlnpqwm?usp=sharing), which is organised as follows (wherever a zipped result file is missing, it means we took the results from the relevant paper):

- **ablation**: files for our ablation experiments
  - **gt, ts, ts_ah, ts_ah_fid, ts_ah_fid_cf_sep**: models and results for students trained with GT, TS, TS+AH, TS+AH+Fidelity, and in non-multitask learning approach (as described in the report)
    - **\*.pth.tar**: the model checkpoint files
    - **results_otb.zip**: raw results on OTB100
- **architecture**: files for our architecture comparison experiment
  - **medium, small, tiny**: models and results for students with backbones ResNet-18-medium, ResNet-18-small and ResNet-18-tiny, as described in the report
    - **\*.pth.tar**: the model checkpoint files
    - **results_otb.zip**: raw results on OTB100

- **atom_default, drnet_default**: models and results for the default [ATOM](https://openaccess.thecvf.com/content_CVPR_2019/html/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.html) and [DRNet](https://openaccess.thecvf.com/content_ICCVW_2019/html/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.html)
  - **\*.pth**: the model checkpoint files
  - **results_otb.zip**: raw results on OTB100
  - **results_lasot_trackingnet.zip**: raw results on LaSOT and TrackingNet
  - **results_vot2018.zip**: raw results on VOT2018 (only for DRNet)

- **atom_cfkd**: model and results for the student ATOM with MobileNetV2 backbone trained with our CFKD method
  - **atom_cfkd.pth.tar**: the model checkpoint file
  - **results_otb.zip**: raw results on OTB100
  - **results_lasot_trackingnet.zip**: raw results on LaSOT and TrackingNet
  - **results_vot2018.zip**: raw results on VOT2018
  - **results_vot2019.zip**: raw results on VOT2019

- **atom_compression, atom_tskd**: models and results for the student ATOM with MobileNetV2 backbone trained using [Wang's](https://ieeexplore.ieee.org/abstract/document/9080535?casa_token=iOe3fxsyvN4AAAAA:BuEYvFcYvTIgBVWtqkKlokDs_D1WiGYnuMQaQXKo7aT2cV9kYyhaGNbTtJKIuBWwgL0Y5CgBpA) and [Liu's](https://arxiv.org/abs/1907.10586) methods
  - **\*.pth.tar**: the model checkpoint files
  - **results_otb.zip**: raw results on OTB100
  - **results_lasot_trackingnet.zip**: raw results on LaSOT and TrackingNet

- **drnet_cfkd**: model and results for the student DRNet with MobileNetV2 backbone trained with our CFKD method
  - **drnet_cfkd.pth.tar**: the model checkpoint file
  - **results_otb.zip**: raw results on OTB100
  - **results_lasot_trackingnet.zip**: raw results on LaSOT and TrackingNet
  - **results_vot2018.zip**: raw results on VOT2018
  - **results_vot2019.zip**: raw results on VOT2019



## Acknowledgement
We use the [PyTracking](https://github.com/visionml/pytracking) framework developed by Danelljan et al. While the overall framework and source code is kept unchanged, we add loss functions, actors, and trainer implementations to fulfil our purpose.

## Citation
```
@MastersThesis{Duolikun:2020,
    author    =     {Danier Duolikun},
    title     =     {{Knowledge Distillation for Discriminative Correlation Filter Trackers}},
    school    =     {Univesity College London},
    address   =     {United Kingdom},
    year      =     {2020},
}
```
