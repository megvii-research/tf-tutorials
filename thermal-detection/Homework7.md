## Homework 7
### Goals
- Design a forehead detector in each human face in thermal images.

### Tasks
Locating the forehead region(naked skin between hair and eyebrows) accurately is an important stage in human temperature measurement applications. Since there are much less thermal images labeled with forehead regions, we need to find an appropriate method to train neural networks with few or no thermal training images, though it will be easy to acquire some RGB face images for training.

#### Q1. Find your own way to design the forehead detector
One suggestion is that you can synthesize thermal images, construct your own training set and train your detector. You may also explore other methods such as using face landmarks to calculate the forehead region and use this pipeline as if it is a detector.

#### Q2. Calculate IoU of your detector
In this experiment, we will provide a test set with 131 images, each labeled with bounding box of faces and foreheads.  The test set will be distributed through TA. Run your detector and calculate the average Intersection over Unions(IoU) with the labeled bounding box in each image.

### Other instructions
- You can use [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ) with segmentation mask to synthesize the training images and generate the bounding boxes of foreheads with segmentation masks of eyebrows and hair.
- You can run view.py to visualize the training set. Thermal image is shown as a grayscale one .The green box bounds the face area and the red box bounds the forehead region. 
<img src="./label_thermal.png" width="380px"/>



