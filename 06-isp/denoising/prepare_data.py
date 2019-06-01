import os
import cv2
import numpy as np
import glob
import argparse

test_dataset_path = '../CBSD68/CBSD68'
test_list = glob.glob(test_dataset_path + '/*.png')
dataset_meta = (test_list, len(test_list))

def main(args):
    target_path = os.path.join('../CBSD68','CBSD68_{}'.format(args.noise_level))
    if not os.path.exists(target_path):
        os.system('mkdir {}'.format(target_path))
    for i in range(len(test_list)):
        img_path = test_list[i]
        img = cv2.imread(img_path)
        noisy_img = img.astype(np.float32)/255.0 + np.random.normal(0, args.noise_level/255.0, img.shape)
        noisy_img_path = os.path.join(target_path, img_path.split('/')[-1])
        cv2.imwrite(noisy_img_path,(noisy_img*255).astype(np.float32))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_level',type = int)
    args = parser.parse_args()
    main(args)

