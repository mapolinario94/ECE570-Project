# Analog to Spiking Neural Network Conversion
# ECE57000: Artificial Intelligence (Fall 2021) - Course Project

## Abstract:
Spiking neural networks have recently emerged as promising bio-realistic neural models 
for machine learning tasks. Some features like sparsity, event-driven processing, and 
spatio-temporal representation can potentially be exploited to bring intelligence to 
edge devices. However, deep spiking models remain challenging to train mainly because 
their activation function is not differentiable, so the errors can not be 
back-propagated directly. In this work, we explore the analog to spiking conversion 
approach to train spiking models. Specifically, we analyze how the reset mechanism 
plays a central role in the conversion process. Moreover, we perform experiments in a 
low-latency regime using three different ANN models, a small CNN, VGG5, and VGG16, 
achieving near-lossless (<1%) accuracy conversion.

## Implementation:
This work partially reproduces some results about soft-reset mechanisms in spiking 
neural networks shown in *Han et al. (2020)*. 

This repository has the following structure:

- **./models:**
  - **./models/spiking_layers.py:** implementation of spiking version of nn.Linear 
    and nn.Conv2d classes from Pytorch. In this script, we reuse the implementation 
    of *LinearSpike* class from https://github.com/nitin-rathi/hybrid-snn-conversion 
    to implement the Heaviside function with a piecewise-linear surrogate gradients.
  - **./models/spiking_models.py:** implementation of a class SpikingModel which 
    inherit from nn.Module, it is intented to contain to nn.Sequential objects 
    (features and classifier) instances of spiking layers. 
  - **./models/conversion_method.py:**  implementation of the SpikeNorm algorithm 
    proposed in *Sengupta et al. (2019)*.
- **./utils:**
  - **./utils/dataloader.py:** implementation of a dataloader function to manage 
    the CIFAR10 dataset.
  - **./utils/metrics.py:**  implementation of two functions *test* and 
    *timesteps_performance* to evaluate the model in the CIFAR10 test set. First 
    function eval the model in one step, while the second function evaluate the 
    model recursively with multiple time steps and threshold values.
- **./notebooks:** all the experiments were implemented in Jupyter notebooks. See 
  each notebooks for more details.
  - **./notebooks/VGG16_conversion.ipynb**: experiments with soft-reset for VGG16.
  - **./notebooks/VGG5_conversion.ipynb**: experiments with soft and hard reset for 
    VGG5.
  - **./notebooks/simple_spiking_model.ipynb**: experiments for a small CNN.

## How to run:
This work used CIFAR10 obtained from PyTorch. Follow the next steps to run the 
experiments:
1. Create a Python environment. The main libraries required are:
  - torch==1.9.0+cu111
  - torchaudio==0.9.0
  - torchtext==0.10.0
  - torchvision==0.10.0+cu111
  - torchviz==0.0.2
  - matplotlib==3.4.3
  - jupyter==1.0.0
2. Download the file pre_trained_models from the following link: https://purdue0-my.sharepoint.com/:f:/g/personal/mapolina_purdue_edu/EiUkMDTMEZNIrcMFIFXjLpEBeWa_njsFo_2gvLJIvcssjw?e=0wWxRV 
   (Note: the original pre-trained models for ANN VGG5 and VGG16 where obtained from 
   https://github.com/nitin-rathi/hybrid-snn-conversion , but we modified their 
   structure to be compatible with out implementation).
   
3. Initiate a Jupyter server. Then, run the notebooks provided in ./notebooks file. (
   Note: VGG5 and VGG16 could take several hours to complete
   )

## Reference:
[1] Han, B., Srinivasan, G. and Roy, K. *"RMP-SNN: Residual Membrane Potential Neuron for 
Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network"*. In 2020 IEEE/CVF 
Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 13555-13564, 
doi:10.1109/CVPR42600.2020.01357.

[2] Rathi, N., Srinivasan, G., Panda, P., and Roy, K. *"Enabling deep spiking 
neural networks with hybrid con-version and spike timing dependent backpropagation"*. 
In 2020 International Conference on Learning Representations,  2020.

[3] Li, Y., Deng, S., Dong, X., Gong, R., and Gu, S. *"A free 
lunch from ANN: Towards efficient, accurate spiking neural networks calibration"*. In 
International Conference on Machine Learning (ICML), 2021.

[4] Sengupta, A., Ye, Y., Wang, R., Liu, C., and Roy, K. *"Going deeper in spiking 
neural networks: VGG  and residual architectures"*. Frontiers in Neuroscience, 
13:95, 2019. ISSN 1662-453X. doi:10.3389/fnins.2019.00095
