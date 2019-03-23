# Homework 1a
## Goals
- Learn how the distribution of training data influences the results.
- Learn how various data augmentation techniques help when the size of the training dataset is small.

## Requirements
- The work has to be finished individually. Plagiarism will be dealt with seriously.
- In this experiment, we only use data from training data file 'train32x32.mat'. Please always set the "use_extra_data" flag to **False** in common.py.

## Tasks
- #### Q1: How a smaller dataset affects test accuracy?
  - a. Feed all the data from train32x32.mat.
  - b. Feed 30000 images from train32x32.mat.
  - c. Feed 10000 images from train32x32.mat.
  
  **_Submit your code in q1.a.diff q1.b.diff and q1.c.diff respectively._**
   
- #### Q2: How the distribution of data affects test accuracy?
   - Reduce the size of training data in the following ways, 
     - reduce the amount of images labelled with'8' '9' and '0' to 500 and get Dataset A

       cnt = [500, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 500, 500]
     - reduce the amount of images labelled with '6', '7', '8','9' and '0' to 1000 and get Dataset B

       cnt = [1000, 13861, 10585, 8497, 7458, 6882, 1000, 1000, 1000, 1000]
     - reduce the amount of images labelled with '1','2','3','4'and '5' to 6000 and get Dataset C

       cnt = [4948, 6000, 6000, 6000, 6000, 6000, 5727, 5595, 5045, 4659]

  - Rerun the classification task with Dataset A, Dataset B and Dataset C respectively. Compare your results with those from Q1.a.
    
    **_Submit your code in q2.a.diff q2.b.diff q2.c.diff respectively._**
    
    _Note_: You might need to shuffle the data after contructing the datasets to maintain the distribution for experiments with subsets from these datasets.
  - Rerun the classification task with 30000 images from Dataset A,  Dataset B and Dataset C respectively. Compare your results with those from Q1.b.
    
    **_Submit your code in q2.{dataset_index}.{dataset_size}.diff, e.g. q2.a.30000.diff if you get 30000 images from Dataset A._**
  - Rerun the classification task with 10000 images from Dataset A,  Dataset B and Dataset C respectively. Compare your results with those from Q1.c.
   
   **_Submit your code in q2.{dataset_index}.{dataset_size}.diff, e.g. q2.b.10000.diff, respectively._**
- #### Q3: How augmentation helps when training dataset is small?
   - Implement some of the augmentation techniques described below.
      * a. color inversion: sets a pixel value from v to 255-v. 
        
       _Note_: You may invert pixels from each channel individually with a probability
      * b. affine transformation: affine transformation is usually adopted for expressing rotations, translations and scale operations.  You may check [here](https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html) to find out more about this augmentation in OpenCV document.
      * c. adding salt and pepper noise: sets a pixel value to 255(make it white-ish as "salt") or to 0( make it black-ish as "pepper")
      
      Apply these augmentation techniques when you feed all/30000/10000 images from train32x32.mat and see whether the test accuracy increases compared with that from Q1.
      
      **_Submit your code in q3.{augmentation-index}.{dataset-size}.diff, e.g. q3.a.10000.diff for color inversion augmentation applied to dataset size of 10000._**
  - (Optional) You might also try combinations of augmentation techniques or other augmentation techniques and see whether it helps.
      
      **_Pick up at most two augmentations that helps and submit your code in q3.o1.diff, q3.o2.diff etc..._**

- #### Q4: Whether [_Mixup_]((https://arxiv.org/abs/1710.09412))(a data-agnostic augmentation technique) helps
    - Apply mixup technique when you feed all/30000/10000 images from train32x32.mat in the classification tasks and see whether the test accuracy increases compared with that from Q1.
      
      **_Submit your code in q4.{dataset-size}.diff, e.g. q4.all.diff or q4.10000.diff._**

## Summary of diff file


|Task | File name|
|:---:|:---:| 
|Q1|q1.a.diff q1.b.diff q1.c.diff
|Q2|q2.a.diff q2.b.diff q2.c.diff 
|| q2.a.30000.diff q2.b.30000.diff q2.c.30000.diff
||   q2.a.10000.diff q2.b.10000.diff q2.c.10000.diff
|Q3| q3.a.all.diff    q3.b.all.diff  q3.c.all.diff 
||   q3.a.30000.diff  q3.b.30000.diff  q3.c.30000.diff 
 ||  q3.a.10000.diff  q3.b.10000.diff  q3.c.10000.diff
||(Optional)q3.o1.diff q3.o2.diff
|Q4|q4.all.diff q4.30000.diff q4.10000.diff


