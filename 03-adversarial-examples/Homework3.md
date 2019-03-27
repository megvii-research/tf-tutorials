## Homework 3

### Goals
 * Learn the basic principle of generating adversarial examples.
### Background
By adding small perturbations to an image, we can force a wrong classification of a trained neural network. The problem is formulated as, finding <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{x}" title="\widehat{x}" /></a>
s.t. <a href="https://www.codecogs.com/eqnedit.php?latex=D(x,&space;\widehat{x})&space;\leq&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D(x,&space;\widehat{x})&space;\leq&space;\epsilon" title="D(x, \widehat{x}) \leq \epsilon" /></a>, 
and <a href="https://www.codecogs.com/eqnedit.php?latex=$argmax_{j}P(y_j|\hat{x},&space;\theta)&space;\neq&space;y_{true}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$argmax_{j}P(y_j|\hat{x},&space;\theta)&space;\neq&space;y_{true}$" title="$argmax_{j}P(y_j|\hat{x}, \theta) \neq y_{true}$" /></a> (untargeted),
or <a href="https://www.codecogs.com/eqnedit.php?latex=argmax_{j}P(y_j|\hat{x},&space;\theta)&space;=&space;y_{target}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?argmax_{j}P(y_j|\hat{x},&space;\theta)&space;=&space;y_{target}" title="argmax_{j}P(y_j|\hat{x}, \theta) = y_{target}" /></a>(targeted), where <a href="https://www.codecogs.com/eqnedit.php?latex=P(y|x,\theta)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(y|x,\theta)" title="P(y|x,\theta)" /></a> is a classifier model parameterized by <a href="https://www.codecogs.com/eqnedit.php?latex=$\theta$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\theta$" title="$\theta$" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=$\epsilon$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\epsilon$" title="$\epsilon$" /></a> is the maximum allowed noise. And <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{x}" title="\widehat{x}" /></a> is the generated adversarial examples.

* In this experiment we target 100 images from [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and use a VGG-like model.
* We evaluate the generated examples with perturbation scale and success rate, i.e. the  probability of generated adversarial examples being misclassified by the model.


### Tasks

- #### Q1: Generating untargeted adversarial examples
 Run the baseline code and record the perturbation scale and success rate.

- #### Q2: Adding loss terms
   - Adding averaged total variation loss(explained in Homework2) and see how the perturbation scale and success rate changes.
     #### Submit your code in q2.1.diff
  - To obtain minimum distortion, add l2 loss between adversarial examples and the original images.
    You may construct your loss as，
<a href="https://www.codecogs.com/eqnedit.php?latex=$L&space;=&space;crossentropyloss(preds,&space;gt)&space;&plus;&space;c(\widehat{x}&space;-&space;x)^2$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$L&space;=&space;crossentropyloss(preds,&space;gt)&space;&plus;&space;c(\widehat{x}&space;-&space;x)^2$" title="$L = crossentropyloss(preds, gt) + c(\widehat{x} - x)^2$" /></a>
Rerun the experiment to see how success rate changes with different c value, e.g. 1, 0.1, 0.01 etc.
    #### Choose one c value and submit your code in q2.2.diff
- #### Q3: Whether augmentation helps defending adversarial attacks?
  - Implement one or some of the augmentation techniques, e.g. affine transformation, adding salt and pepper noise, [bluring](https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html) etc. on the generated adversarial examples. Evaluate the augmented adversarial examples and find out whether these examples still get misclassified.
    #### Pick up one augmentation or combination that helps and submit your code in q3.diff.
- #### Q4: Generating targeted adversarial examples
  - In targeted attack scenario, please assign a target label for each image. After training, the corresponding examples will be classified as this label. You need to modify the loss term and rerun the generating process. You may also increase the number of epochs in each run.
     #### Submit your code in q4.diff.


### Appendix: literature you may be interested in
- I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and Harnessing Adversarial examples. In International Conference on Learning Representations, 2015.
- N. Carlini and D. Wagner. Towards evaluating the robustness of neural networks. In IEEE Symposium on Security and Privacy (SP), pages 39–57, 2017
- S. M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard. Deepfool:a simple and accurate method to fool deep neural networks.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2574–2582, 2016.

