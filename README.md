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
neural networks shown in **Han et al. (2020)**. 

- ./models:
  - ./models/spiking_layers.py:
  - ./models/spiking_models.py:
  - ./models/conversion_method.py:  
- ./utils:
  - ./utils/dataloader.py
  - ./utils/metrics.py:  
- ./notebooks:
  - ./notebooks/

## Reference:
[1] B. Han, G. Srinivasan and K. Roy, *"RMP-SNN: Residual Membrane Potential Neuron for 
Enabling Deeper High-Accuracy and Low-Latency Spiking Neural Network,"* 2020 IEEE/CVF 
Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 13555-13564, 
doi:10.1109/CVPR42600.2020.01357.
