# Homework 2
## Goals
- Learn the principle of [representation inverting](https://arxiv.org/abs/1412.0035), i.e. approximate an image with its feature extracted from deep neural networks.

## Backgrounds
The representation inverting problem is formulated as finding an image $x$ whose representation, e.g. extracted features best match the one given.

The representation inverting problem is formulated as finding an image whose representation, e.g. extracted features, best matches the given target， as <img src="https://latex.codecogs.com/gif.latex?$x&space;=&space;argmin_{x&space;\in&space;\mathbb{R}^{H&space;\times&space;W&space;\times&space;C}}L(\Phi(x)-\Phi_0)&space;&plus;&space;\lambda&space;R(x)$" title="$x = argmin_{x \in \mathbb{R}^{H \times W \times C}}L(\Phi(x)-\Phi_0) + \lambda R(x)$" />
where the loss L compares the representation of image x and that of the target <img src="https://latex.codecogs.com/gif.latex?$\Phi_0$" title="$\Phi_0$" /> and R serves as the regularization term. 
* In this experiment we generate the representation using a pretrained vgg16 model with batch normalization added after each convolution layer from the [onnx model zoo](https://github.com/onnx/models). 
* We use L2 loss to compare the difference between representations.
* During the training process, the image $x$ gets tuned and saved per 10 epochs. 
Check the stored images and find out how well this image approximates the target.



## Tasks
- #### Q1: How learning rate helps?
  - The baseline takes constant learning rate 1e-3. Modify the learning rate curve and pick one that achieves better convergence.
    You may use this configuration as your baseline.
  
  **_Submit your code in q1.diff._**
   
- #### Q2: How regularization term helps?
   - Total variation is defined as <img src="https://latex.codecogs.com/gif.latex?$&space;\sum\nolimits_{i,j}&space;(y_{i&plus;1,j}-y_{i,j})^2&space;&plus;&space;(y_{i,j&plus;1}-y_{i,j})^2$" title="$ \sum\nolimits_{i,j} (y_{i+1,j}-y_{i,j})^2 + (y_{i,j+1}-y_{i,j})^2$" />
    where i,j denotes row and column index of a pixel in the image.
Total variation loss encourages images to be made up of piece-wise constant patches. 
Add total variation loss in the loss term and compare the generated images with that in Q1. 
     
   **_Submit your code in q2.1.diff._**
  - Increase the portion of total variation loss in the loss term and rerun the training process.
   
   **_Submit your code in q2.2.diff._**
  - Increase the portion of feature/representation loss and rerun the training process.     
   
   **_Submit your code in q2.3.diff._**
  
- #### Q3: Which is more helpful, shallow or deep representation?
  - Use the features generated from 'conv1_1' and construct the representation loss. Rerun the training process.

  **_Submit your code in q3.1.diff._**
  - Use the features generated from 'pool5' and construct the representation loss. Rerun the training process.
   
  **_Submit your code in q3.2.diff._**
  - Construct representation loss from features from multiple layers. e.g. features from 'conv3_1' and 'fc6' and rerun the training process. 

  **_Submit your code in q3.3.diff._**

- #### Q4: What does different feature extraction model bring?
  - Change the feature extraction model from pretrained VGG16 model to [Resnet18 model](https://pytorch.org/docs/stable/torchvision/models.html).
    Use the feature generated from second residual block , i.e. 'res2' to construct representation loss and rerun the training process.
   You may also play with other layes, e.g. the first convolution layer in each residual block, e.g.'middle1'.

   **_Choose one layer configuration and submit your code in q4.diff._**


## Other instructions
- Please download the parameters of the pretrained VGG16 models [here](https://pan.baidu.com/s/1a59ZJBv2CoGZdhX45o1pqg)(verification code:31w3).
- Please download the parameters of the pretrained Resnet18 models [here](https://pan.baidu.com/s/1ZhxAzkEuplecdzOkv-X0RQ)(verification code:d8rk).
- You can use the image provided in the project as the target or use your own image. It is suggested that your own image is larger than 224x224.

