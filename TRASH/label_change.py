# pyinstaller -F -w bbox_labeler.py
# pyinstaller --onefile -w bbox_labeler.py
# import binascii
import ast
import csv
from glob import glob
import json
import os
from tkinter import *
from tkinter import filedialog, messagebox
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


DEBUG = 1
AUTO_RUN = 1
BG = 2
SAVE_IMG = 1
# DRAW_CNT = 1
MAKE_DENOISE_LABEL = 1
CONVERT_BBOX = 1

## -------------------------------------------------------------------------------- RUN
IDX, bbox_list, BBOX_CNT = 0, [], 0
h, w = 80, 80
SIZE = 12
re_size = (h*SIZE, w*SIZE)

error = 239
cnt_thresh = 1
j = 0
new_label = 0

BG_LIST = []
BG_LENGTH = 40

bboxIdList, bboxId = [], None
hl, vl = None, None
image = None

df = pd.DataFrame()
save_dir = f"out"
save_csv_path = f"{save_dir}/output.csv"
data_dir = f"/media/z/0/MVPC10/DATA/v1.1/RAW/03"

## -------------------------------------------------------------------------------- DOT
DOT = 0


# dot_csv = f"../dot2.csv"
# dot_df = pd.read_csv(dot_csv)
# dot_list = []
# for i in range(len(dot_df.index)):
#     dot_list.append([int(dot_df.iloc[i,0]), int(dot_df.iloc[i,1])])
# print(dot_list)

def dot_mkr(h, w, x, y):
    li = []
    for i in range(x//2, h, x):
        for j in range(y//2, w, y):
            li.append([i, j])
    return li


dot_list = dot_mkr(h, w, 4, 4)
print(dot_list)

## -------------------------------------------------------------------------------- PIXEL LABEL
pixel_thresh = 24


def real_list(li):
    C = [[16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 22, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 31, 31, 32, 32, 33, 34, 34, 35, 35, 36, 37, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 43, 44, 44, 45, 46, 46, 47, 47, 48, 49, 49, 50, 50, 51, 52, 52, 53, 53, 54, 55, 55, 56, 56, 57, 58, 58, 59, 59, 60, 61, 61, 62, 62, 63, 64, ],
         [19, 19, 20, 21, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 33, 34, 34, 35, 36, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 45, 46, 47, 48, 48, 49, 49, 50, 51, 51, 52, 53, 54, 55, 55, 56, 57, 58, 59, 59, 60, 61, 62, 63, 63, 64, 64, 65, 66, 66, 67, 68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79, ]]
    real_list = []
    for i in li:
        x1 = C[0][i[0]]
        y1 = C[1][i[1]]
        x2 = C[0][i[2]]
        y2 = C[1][i[3]]
        real_list.append([x1, y1, x2, y2])

    return real_list


def list_sort(arr):
    padded_list = [None]*80
    for i in arr:
        try:
            if i[1] > 0:
                if padded_list[i[1]] == None:
                    padded_list[i[1]] = [i]
                else:
                    padded_list[i[1]].append(i)
        except Exception as E:
            pass

    sorted_list = []
    for i in padded_list:
        if i != None:
            if len(i) == 1:
                sorted_list.append(i[0])
            else:
                for j in i:
                    sorted_list.append(j)

    return sorted_list


def pixel_label_mkr(img, bbox_label, thresh):
    bbox_label1 = real_list(bbox_label)
    sorted_list = list_sort(bbox_label1)
    pixel_label = np.zeros((w, h), dtype=np.uint8)
    for i in sorted_list:
        pixel_label[i[1]:i[3], i[0]:i[2]] = (img[i[1]:i[3], i[0]:i[2]] > thresh)*1

    return pixel_label, bbox_label1


def bg_filter(target):
    BG_8 = BG_LIST[8:]
    bg = np.zeros([h, w], dtype=int)
    for bg_arr in BG_8:
        bg += bg_arr
    bg //= len(BG_8)

    img = target-bg

    low = 0
    img[img < low] = 0
    img -= img.min()

    return img.astype(np.uint8)


def insert_to_df():
    global bbox_list
    bbox_len = len(bbox_list)
    if bbox_len == 0:
        # bbox_list = []
        bbox_list.append([0, 0, 0, 0])
    elif bbox_len > 1:
        if bbox_list[0] == [0, 0, 0, 0] or bbox_list[0] == 0 or bbox_list[0] == -1:
            bbox_list.pop(0)
    json_loc = json.dumps(bbox_list)
    df.iloc[IDX, 1] = json_loc
    # df.iloc[IDX, 2] = BBOX_CNT
    # df.iloc[IDX, 2] = df.iloc[IDX, 2].astype(int)


def read_data(n):
    global IDX, ir, rgb, data, bbox_list, BBOX_CNT
    clear()
    IDX += n
    data = df.iloc[IDX, 0]
    try:
        # print(df.iloc[IDX, 1])
        bbox_list = json.loads(df.iloc[IDX, 1])
        # if CONVERT_BBOX == 1:
        #     bbox_list1 = real_list(bbox_list)

        # for i, l in enumerate(bbox_list):
        #     bbox_list[i] = [l[0], l[1]+16, l[2], l[3]+20]
        # print(bbox_list)
    except Exception as E:
        bbox_list = []
        log(f"[!!] {E}: {IDX}")
        traceback.print_exc()
        pass
        # log(f"{TE}: {IDX}")
        # traceback.print_exc()
    # BBOX_CNT = len(bbox_list)
    # if bbox_list[0] == -1:  change(1)
    try:
        ir = glob(f"{data_dir}/**/{data}.png")[0]
        rgb = glob(f"{data_dir}/**/{data}.jpg")[0]
    except Exception as E:
        log(f"[!!] {E}: {IDX}")
        traceback.print_exc()

    return IDX


def change(n):
    IDX = read_data(n)

    image1_text.set(f"{rgb}")
    image2_text.set(f"{ir}")

    try:
        if open_new_img():
            index.insert(0, IDX)
        else:
            index.insert(0, f"{IDX}: ERROR!")
    # except FileNotFoundError as FE:
    #     log(f"{FE}: {IDX}")
    #     traceback.print_exc()
    # except IndexError as IE:
    #     log(f"{IE}: {IDX}, FINISHED")
    #     traceback.print_exc()
    #     IDX -= n
    # except TypeError as TE:
    #     log(f"{TE}: {IDX}")
    #     traceback.print_exc()
    except Exception as E:
        log(f"[!!] {E}: {IDX}")
        traceback.print_exc()
        # trace_back = traceback.format_exc()
        # print(f'{E}{chr(10)}{trace_back}')
        # pass

    if AUTO_RUN == 0:
        img = resized_image2.copy()
        draw_txt(img, IDX, side=1)



df = pd.read_csv(file)
df.sort_values(by=df.keys()[0], inplace=True, ascending=True)

