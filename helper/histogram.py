import numpy as np


def imgArray_to_histogramArray(img_arr):
    total_pixel = 0
    hist_arr = np.zeros((np.max(img_arr)+1), dtype=int)
    for row in img_arr:
        for value in row:
            hist_arr[value] += 1
            total_pixel += 1
    return hist_arr

