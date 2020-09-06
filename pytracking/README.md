# PyTracking

A general python library for visual tracking algorithms. 
## Table of Contents

* [Running a tracker](#running-a-tracker)
* [Overview](#overview)
* [Trackers](#trackers)
   * [ATOM](#ATOM)
   * [DRNet](#DRNet)
* [Analysis](#analysis)
* [Libs](#libs)
* [VOT Integration](#vot-integration)


## Running a tracker
The installation script will automatically generate a local configuration file  "evaluation/local.py". In case the file was not generated, run ```evaluation.environment.create_default_local_file()``` to generate it. Next, set the paths to the datasets you want
to use for evaluations. You can also change the path to the networks folder, and the path to the results folder, if you do not want to use the default paths. If all the dependencies have been correctly installed, the you then need to download the models from [here](https://drive.google.com/drive/folders/19mDhiPbQlxCUtnB-qy536RMQ6mlnpqwm?usp=sharing) and place them in ```networks``` folder, and make sure the name of the checkpoint files in the [parameter files](parameter) are correct.

**Note**: to evaluate our distilled trackers, the user needs to download the model checkpoint files from the shared [folder](https://drive.google.com/drive/folders/19mDhiPbQlxCUtnB-qy536RMQ6mlnpqwm?usp=sharing) and put them in the ```network``` folder that will be generated after running the installation script.

**Run the tracker on some dataset sequence**  
This is done using the run_tracker script. 
```bash
python run_tracker.py tracker_name parameter_name --dataset_name dataset_name --sequence sequence --debug debug --threads threads
```  

Here, the dataset_name is the name of the dataset used for evaluation, e.g. ```otb```. See [evaluation.datasets.py](evaluation/datasets.py) for the list of datasets which are supported. The sequence can either be an integer denoting the index of the sequence in the dataset, or the name of the sequence, e.g. ```'Soccer'```.
The ```debug``` parameter can be used to control the level of debug visualizations. ```threads``` parameter can be used to run on multiple threads.

**Run the tracker on a set of datasets**  
This is done using the run_experiment script. To use this, first you need to create an experiment setting file in ```pytracking/experiments```. See [myexperiments.py](experiments/myexperiments.py) for reference. 
```bash
python run_experiment.py experiment_module experiment_name --dataset_name dataset_name --sequence sequence  --debug debug --threads threads
```  
Here, ```experiment_module```  is the name of the experiment setting file, e.g. ```myexperiments``` , and ``` experiment_name```  is the name of the experiment setting.

## Overview
The tookit consists of the following sub-modules.  
 - [analysis](analysis): Contains scripts to analyse tracking performance, e.g. obtain success plots, compute AUC score. It also contains a [script](analysis/playback_results.py) to playback saved results for debugging.
 - [evaluation](evaluation): Contains the necessary scripts for running a tracker on a dataset. It also contains integration of a number of standard tracking and video object segmentation datasets.  
 - [experiments](experiments): The experiment setting files must be stored here,  
 - [features](features): Contains tools for feature extraction, data augmentation and wrapping networks.  
 - [libs](libs): Includes libraries for optimization, dcf, etc.  
 - [notebooks](notebooks) Jupyter notebooks to analyze tracker performance.
 - [parameter](parameter): Contains the parameter settings for different trackers.  
 - [tracker](tracker): Contains the implementations of different trackers.  
 - [util_scripts](util_scripts): Some util scripts for e.g. generating packed results for evaluation on GOT-10k and TrackingNet evaluation servers, downloading pre-computed results. 
 - [utils](utils): Some util functions. 
 - [VOT](VOT): VOT Integration.  
 
## Trackers
 The toolkit contains the implementation of the following trackers.

 **Note**: 
    
    1. Make sure the checkpoint files in the networks folder have the same name as the file names used in the parameter files.
    2. For evaluation on CPU, first use this [script](../create_cpu_model.py) to create CPU-compatible models, then set the "use_gpu" parameter in the parameter files to False.
### ATOM
The official implementation for the ATOM tracker ([paper](https://arxiv.org/abs/1811.07628)). 
The tracker implementation file can be found at [tracker.atom](tracker/atom).  
 
#### Parameter Files
* **[default](parameter/atom/default.py)**: The default parameter setting that was used to produce all ATOM results in the original paper, except on VOT.  
* **[default_vot](parameter/atom/default_vot.py)**: The parameters settings used to generate the VOT results.  
* **[mobilenetsmall](parameter/atom/mobilenetsmall.py)**: The parameters for student ATOM tracker with MobileNetV2 backbone, trained with any method.
* **[mobilenetsmall_vot](parameter/atom/mobilenetsmall.py)**: The VOT parameters for the student ATOM tracker with MobileNetV2 backbone.
* **[resnet18medium](parameter/atom/resnet18medium.py)**: The parameters for student ATOM tracker with ResNet-18-medium backbone.
* **[resnet18small](parameter/atom/resnet18small.py)**: The parameters for student ATOM tracker with ResNet-18-small backbone.
* **[resnet18tiny](parameter/atom/resnet18tiny.py)**: The parameters for student ATOM tracker with ResNet-18-tiny backbone.

### DRNet
The original implementation for the DRNet tracker is provided here([repository](https://github.com/ShuaiBai623/DRNet)).  
 
#### Parameter Files
* **[default](parameter/atom/default.py)**: The default parameter setting that was used to produce DRNet results.
* **[mobilenetsmall](parameter/atom/mobilenetsmall.py)**: The parameters for student DRNet tracker with MobileNetV2 backbone, trained with any method.


## Analysis  
The [analysis](analysis) module contains several scripts to analyze tracking performance on standard datasets. It can be used to obtain Precision and Success plots, compute AUC, OP, and Precision scores. The module includes utilities to perform per sequence analysis of the trackers. Further, it includes a [script](analysis/playback_results.py) to visualize pre-computed tracking results. 

For example, to plot the success and precision curves for OPE on OTB100 of an ATOM tracker, making sure that the results are in the directory ```tracking_results/atom/<parameter file>```, one can run the following code
```python
import os
import sys
import matplotlib.pyplot as plt
os.chdir('pytracking/pytracking/analysis')
sys.path.append('../..')

from pytracking.analysis.plot_results import plot_results, print_results
from pytracking.evaluation import Tracker, get_dataset, trackerlist

trackers = []
trackers.extend(trackerlist('atom', '<parameter file>', None, 'atom_cfkd'))

dataset = get_dataset('otb')

plot_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec'), skip_missing_seq=True, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
```

## Libs
The pytracking repository includes some general libraries for implementing and developing different kinds of visual trackers, including deep learning based, optimization based and correlation filter based. The following libs are included:

* [**Optimization**](libs/optimization.py): Efficient optimizers aimed for online learning, including the Gauss-Newton and Conjugate Gradient based optimizer used in ATOM.
* [**Complex**](libs/complex.py): Complex tensors and operations for PyTorch, which can be used for DCF trackers.
* [**Fourier**](libs/fourier.py): Fourier tools and operations, which can be used for implementing DCF trackers.
* [**DCF**](libs/dcf.py): Some general tools for DCF trackers.

## VOT Integration
#### Python Toolkit
Install the vot-python-toolkit and set up the workspace, as described in https://www.votchallenge.net/howto/tutorial_python.html. Copy our [VOT/trackers.ini](VOT/trackers.ini) file to the workspace as described in the tutorial.

 
