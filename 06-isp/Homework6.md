# Homework 6
## Goals
- Learn the principle of image denoising/super resolution/demosaicing with deep neural networks.

##  Background and tasks 

### 1.  Denoising

The goal of image denoising is to recover a clean image <img src="https://latex.codecogs.com/gif.latex?x" title="x" /> from a noisy observation <img src="https://latex.codecogs.com/gif.latex?y&space;=&space;x&space;&plus;&space;v" title="y = x + v" />, where <img src="https://latex.codecogs.com/gif.latex?v" title="v" /> is additive noise. We can separate the noise from noisy image using feed-forward convolutional neural networks. The baseline project is based on [DnCNN](https://arxiv.org/abs/1608.03981), which exploits residual learning.  Following most denoising works, we use Gaussian noise to synthesize the training data and testing data, and we adopt [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) between the noisy version and the ground truth to evaluate how well the model works.

#### 1.1 Tasks
1. Train your model for Gaussian noise with specific noise level <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /> = 15, 25 and 50, i.e. noise level is kept the same for training and testing.
2. Train your model for Gaussian noise with blind noise level, i.e. set a range of the noise level for training and testing with a specific level <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /> = 15, 25 and 50.

You may need to change the hyper-parameter of the baseline project, apply augmentation or even change the model in order to obtain higher PSNR. **_Choose the best configuration and submit your code in q1.diff._**

#### 1.2  Other instructions
Part of the datasets for color and grayscale images are available at [here](https://pan.baidu.com/s/11T2Q1qdVpISGT44Ur3PY_g)(verfication code:x351 ). Other testsets can be found [here](https://github.com/cszn/DnCNN/tree/master/testsets). The baseline model is trained with color image dataset CBSD432 and tested with CBSD68. You may also switch to greyscale image datasets as stated in the paper, e.g. Train400 for training and BSD68 for testing.

### 2. Super resolution

Single image super resolution methods aims to recover a high resolution image from a low resolution image.  Deep neural network models provide efficient end-to-end restoration methods, e.g. [SRCNN](https://link.springer.com/chapter/10.1007/978-3-319-10593-2_13), [FSRCNN](https://arxiv.org/abs/1608.00367) and [DRRN](https://ieeexplore.ieee.org/document/8099781), etc. We implement our baseline based on [ESPCN](https://arxiv.org/abs/1609.05158), which changes images to YCbCr color space and and only focus on the luminance channel.

#### 2.1 Tasks

1. Train your model for different sub-sampling ratio.  You may need to change the hyper-parameter of the baseline project, apply augmentation or change the model in order to obtain higher PSNR. **_Choose the best configuration and submit your code in q2.diff._**

 #### 2.2 Other instructions
The training and testing datasets are available at [here](http://vllab.ucmerced.edu/wlai24/LapSRN/) and [here](https://github.com/jbhuang0604/SelfExSR). The model in the baseline project is trained with Train91 and tested on Set14. You can add other training datasets(e.g. General100) and test your model with multiple testsets.

### 3. Demosaicing

Color filter arrays(CFAs) in digital cameras make the sensor in every pixel only records partial spectral information. Demosaicing is the process of inferring the missing information for each pixel. The baseline project exploits residual learning between the bilinear interpolation of the CFA output and the ground truth, and targets the commonly used [Bayer CFA](https://en.wikipedia.org/wiki/Bayer_filter).  For the network architecture, we take [DMCNN-VD](https://arxiv.org/abs/1802.03769) as the reference and drop batch normalization layer for better results.

#### 3.1 Tasks
1. Train your model for sRGB space.
2. Train your model for linear space. 

You may need to change the hyper-parameter of the baseline project, apply augmentation or change the model in order to obtain higher PSNR. **_Choose the best configuration and submit your code in q3.diff._**


 #### 3.2 Other instructions
The training dataset for sRGB is available at [here](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Demosaic/).
The testing dataset for sRGB is available at [here](http://inf.ufrgs.br/~bhenz/datasets.zip).
The training and testing datasets for linear space are available at [here](https://www.microsoft.com/en-us/download/details.aspx?id=52535).
You may use the images in the training and testing lists of the bayer_panasonic folder for training , and test your model with the images in the validation list of the bayer_panasonic folder and all the images in the bayer_canon folder.
