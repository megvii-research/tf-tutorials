#!/usr/bin/env python

# vim: ts=4 sw=4 sts=4 expandtab
import numpy as np

def from_img_to_sub_pixel(img, ratio):
    h, w, c = img.shape

    result = np.zeros((int(h/ratio), int(w/ratio), c * ratio * ratio), dtype = np.uint8)
    for i in range(c):
        for j in range(ratio):
            for k in range(ratio):
                result[:,:, i * ratio * ratio + j * ratio + k] = img[j::ratio,k::ratio,i]

    return result


def from_sub_pixel_to_img(pixel, ratio):
    h, w, c = pixel.shape
    result = np.zeros((h * ratio, w *ratio, int(c /ratio /ratio)), dtype=np.float32)
    for i in range(int(c/ratio/ratio)):
        for j in range(ratio):
            for k in range(ratio):
                result[j::ratio, k::ratio, i] = pixel[:, :, i * ratio * ratio + j * ratio + k]

    return result
