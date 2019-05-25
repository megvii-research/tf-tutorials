# Homework 5
## Goals
- Learn the principle of [texture synthesis using deep neural networks](https://arxiv.org/abs/1505.07376).

## Backgrounds
Textures synthesis can be viewed as approximating a texture image <img src="https://latex.codecogs.com/gif.latex?$x$" title="$x$" /> from randomly initialized noise <img src="https://latex.codecogs.com/gif.latex?$\widehat{x}$" title="$\widehat{x}$" />.  <img src="https://latex.codecogs.com/gif.latex?$\widehat{x}$" title="$\widehat{x}$" /> gets trained and optimized until its features extracted from the deep neural networks have the same statistics as that of the source texture image <img src="https://latex.codecogs.com/gif.latex?$x$" title="$x$" />. 

Since textures are per definition stationary, the statistics of the features should be agnostic to spatial information. In this experiment, we use Gram matrix to describe the spatial-agnostic statistics of the source texture image and the generated texture image. The Gram matrix is defined as the correlation between different channel of features, which is shown as below,

<img src="https://latex.codecogs.com/gif.latex?G_{ij}^l&space;=&space;\sum_{m,n}&space;F_{i,m,n}^l&space;F_{j,m,n}^l" title="G_{ij}^l = \sum_{m,n} F_{i,m,n}^l F_{j,m,n}^l" />

where <img src="https://latex.codecogs.com/gif.latex?$F_{i,m,n}^l$" title="$F_{i,m,n}^l$" /> denotes the pixel with location (m,n) in the ith feature map in layer l.

So we can construct loss function as

<img src="https://latex.codecogs.com/gif.latex?Loss&space;=&space;\sum_l&space;w_l&space;\frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{ij}^l-\widehat{G_{ij}^l})^2" title="Loss = \sum_l w_l \frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{ij}^l-\widehat{G_{ij}^l})^2" />

where <img src="https://latex.codecogs.com/gif.latex?$G&space;\space&space;\widehat{G}$" title="$G \space \widehat{G}$" /> denotes the Gram matrix of the texture image and the generated image, <a href="https://www.codecogs.com/eqnedit.php?latex=N_lM_l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N_lM_l" title="N_lM_l" /></a> is the number of pixels of feature maps in layer l and <img src="https://latex.codecogs.com/gif.latex?w_l" title="w_l" /> is the weighting factor of the contribution of each layer to the total loss.(we simply choose <img src="https://latex.codecogs.com/gif.latex?w_l" title="w_l" /> = 1 for all layers in the baseline project.)

## Other instructions
* In this experiment we still use the pretrained vgg16 model with batch normalization added after each convolution layer from the [onnx model zoo](https://github.com/onnx/models) to extract features as in Homework2. Please download the parameters of the pretrained VGG16 models [here](https://pan.baidu.com/s/1a59ZJBv2CoGZdhX45o1pqg)(verification code:31w3).

* During the training process, the generated image <img src="https://latex.codecogs.com/gif.latex?$\widehat{x}$" title="$\widehat{x}$" /> gets saved every 10 epochs. 
Check the stored images and find out how well it gradually approximates the texture image.

* You can use the image provided in the project as the target or use your own image. It is suggested that your own image is larger than 224x224.



## Tasks
- #### Q1: Implementing Gram matrix and loss function.
  - Use the features extracted from all the 13 convolution layers,  complete the baseline project with loss function based on gram matrix and run the training process.
  **_Submit your code in q1.diff._**
   
- #### Q2: Training with non-texture images.
   - To better understand texture model represents image information, choose another non-texture image(such as robot.jpg in the ./images folder) and rerun the training process. 
**_Submit your code in q2.diff._**
   
- #### Q3: Training with less layers of features.

  - The baseline takes the features from all the convolutional layers, which involves a large number of  parameters. To reduce the parameter size,  please use less layers for extracting features (based on which we compute the Gram matrix and loss) and explore a combination of layers with which we can still synthesize texture images with high degrees of naturalness.
   **_Submit your code with the best configuration in q3.diff._**
 
- #### Q4: Finding alternatives of Gram matrix.
  - We may use the Earth mover's distance between the features of source texture image and the generated image. You may sort the pixel in each feature map of the generated image and the original image and compute their L2 distance to construct the loss, which is shown as below(For rationality of this equivalence please check [here](https://pdfs.semanticscholar.org/194c/2eec28f70ac7da28c7d9f73f65351a181df2.pdf)),

      <img src="https://latex.codecogs.com/gif.latex?$Loss&space;=&space;\sum_l&space;w_l&space;\sum_i&space;(sorted(F_i)-sorted(\hat{F_i}))^2&space;$" title="$Loss = \sum_l w_l \sum_i (sorted(F_i)-sorted(\hat{F_i}))^2 $" /> where <img src="https://latex.codecogs.com/gif.latex?$F_i,&space;\widehat{F_i}" title="$F_i, \widehat{F_i}" /> is the feature map of the original image and the generated image.
      
     **_Submit your code in q4.diff._**

- #### Q5: Training with different weighting factor.
   - Use the configuration in Q3 as baseline. Change the weighting factor of each layer and rerun the training process.
  **_Choose one configuration and submit your code in q5.diff._**
