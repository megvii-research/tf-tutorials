# Homework 1

## Requirement
- The work has to be finished individually. Plagiarism will be dealt with seriously.

## Goals
- Learn to do control experiments
- Learn there are alternatives to Softmax/Cross Entropy when training DNN

## FAQ
#### Q: Where can I get support for this homework?
A: Use "Issues" of this repo.

#### Q: What's the DDL for homeworks?
A: We'll discuss the homework in succeeding experiment course (every two week). Those homeworks turned in after discussion will be capped at 90 marks.

#### Q: How will the score of each homework affect final course score?
A: The algorithm is TBD.

#### Q: I don't have access to a GPU. How do I finish homework in time?
A: You can choose to skip extra\_32x32.mat when trainining. Find a file called common.py in 01-svhn, then set use\_extra\_data = False or True to control.

#### Q: Where to find dataset files?
A: Open http://ufldl.stanford.edu/housenumbers . Please download format2 data. (train\_32x32.mat, test\_32x32.mat, extra\_32x32.mat)

## GPU servers
- We provide some servers, each server has eight Nvidia GPU for students to do all the experiments.
- **python** and **tensorflow** has already been installed.
- Everyone will be assign a IP address and a port number to log in a docker on servers.
- You can read Docker.md for more information.

## Warnings
- **Do not use the computation resource for your private project!!**
- **Do not let anyone who has not selected this course know the username and password of your docker!!**

## Questions
- #### Q1: Finding alternatives of softmax
**(Should be named as q1.1.diff q1.2.diff q1.3.diff q1.4.diff)**

  <img src="./images/find_soft.png" width="500px"/>

- #### Q2: Regression vs Classification **(Should be named as q2.diff)**
  - Change cross entropy loss to the square of euclidean distance between model predicted probability and one hot vector of the true label.

- #### Q3: Lp pooling **(Should be named as q3.diff)**
  - Change all pooling layers to Lp pooling
  - Descriptions about Lp pooling is at https://www.computer.org/csdl/proceedings/icpr/2012/2216/00/06460867.pdf

- #### Q4: Regularization
  - Try Lp regularization with different p. **(Pick one number p with best accuracy and name as q4.1.diff)**
  - Set Lp regularization to a minus number. (L_model + L_reg to L_model - L_reg) **(Should be named as q4.2.diff)**

