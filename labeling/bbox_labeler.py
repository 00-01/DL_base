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
AUTO_RUN = 0
BG = 2
SAVE_IMG = 0
SAVE_PIXEL_LABEL = 0
# DRAW_CNT = 1
MAKE_DENOISE_LABEL = 1
CONVERT_BBOX = 1

## -------------------------------------------------------------------------------- RUN
IDX, bbox_list, BBOX_CNT = 0, [], 0
h, w = 80, 80
SIZE = 12
re_size = (h*SIZE, w*SIZE)

s1, s2 = 400, 400
ss = h*SIZE/s1
s_size = (s1*ss, s2*ss)
s1_mn, s1_mx, s2_mn, s2_mx = s1//4, s1, s2//5, s2-s2//5
# X1, X2, Y1, Y2 = s1_mn, s1_mx, s2_mn, s2_mx
# X1, X2, Y1, Y2 = 103, 348, 104, 309
X1, X2, Y1, Y2 = 103, 354, 104, 312

ERROR = 0
cnt_thresh = 1
j = 0
new_label = 0

BG_LIST = []
BG_LENGTH = 40

COLORS = ['red', 'yellow', 'orange', 'blue', 'pink', 'cyan', 'green']
# initialize mouse state
STATE = {}
STATE['click'] = 0
STATE['x'], STATE['y'] = 0, 0
# reference to bbox
bboxIdList, bboxId = [], None
hl, vl = None, None
image = None

ab_size = 70
a1, b1, a2, b2 = ab_size*2, ab_size, 0, ab_size
x_adj, y_adj = 0, 0
new_img = None
resized_image1, resized_image2 = np.zeros(re_size, dtype=np.uint8), np.zeros(re_size, dtype=np.uint8)

df = pd.DataFrame()
df1 = pd.DataFrame()
save_dir = f"OUT"
save_csv_path = f"{save_dir}/output.csv"
data_dir = f"/media/z/0/MVPC10/DATA/v1.1/RAW/03"

## -------------------------------------------------------------------------------- DOT
DOT = 0
PIXEL = 1
BBOX = 1
BG_RM = 1
DILATE = 0

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

# with open('dot_44.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerow(['x', 'y'])
#     write.writerows(dot_list)

## -------------------------------------------------------------------------------- PIXEL LABEL
pixel_thresh = 15
INSTANCE_SEG = 1


def real_list(li):
    C = [[16, 16, 17, 17, 18, 19, 19, 20, 20, 21, 22, 22, 23, 23, 24, 25, 25, 26, 26, 27, 28, 28, 29, 29, 30, 31, 31, 32, 32, 33, 34, 34, 35, 35, 36, 37, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 43, 44, 44, 45, 46, 46, 47, 47, 48, 49, 49, 50, 50, 51, 52, 52, 53, 53, 54, 55, 55, 56, 56, 57, 58, 58, 59, 59, 60, 61, 61, 62, 62, 63, 64,],
         [19, 19, 20, 21, 21, 22, 23, 24, 25, 25, 26, 27, 28, 29, 29, 30, 31, 32, 33, 33, 34, 34, 35, 36, 36, 37, 38, 39, 40, 40, 41, 42, 43, 44, 44, 45, 46, 47, 48, 48, 49, 49, 50, 51, 51, 52, 53, 54, 55, 55, 56, 57, 58, 59, 59, 60, 61, 62, 63, 63, 64, 64, 65, 66, 66, 67, 68, 69, 70, 70, 71, 72, 73, 74, 74, 75, 76, 77, 78, 78, 79,]]
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


def fill_arr(li):
    flag = 0
    ax0, ax1 = np.shape(li)  ## ax0: 10   ax1: 5
    mn, mx = [-1]*ax1, [-1]*ax1  ## axis 1
    for in0 in range(ax1):
        for in1 in range(ax0):
            if li[in1][in0] == 1:
                if flag == 1:
                    mx[in0] = in1
                elif flag == 0:
                    mn[in0] = in1
                    flag = 1
        # print(mn, mx)
        for i in range(mn[in0], mx[in0]+1):
            if i > 0:
                li[i][in0] = 1
        flag = 0

    mn, mx = [-1]*ax0, [-1]*ax0  ## axis 0
    for in1 in range(ax0):
        for in0 in range(ax1):
            # print(in1, in0)
            if li[in1][in0] == 1:
                if flag == 1:
                    mx[in1] = in0
                elif flag == 0:
                    mn[in1] = in0
                    flag = 1
        # print(mn, mx)
        for i in range(mn[in1], mx[in1]+1):
            if i > 0:
                li[in1][i] = 1
        flag = 0

    mn, mx = [-1]*ax1, [-1]*ax1  ## axis 1
    for in0 in range(ax1):
        for in1 in range(ax0):
            if li[in1][in0] == 1:
                if flag == 1:
                    mx[in0] = in1
                elif flag == 0:
                    mn[in0] = in1
                    flag = 1
        # print(mn, mx)
        for i in range(mn[in0], mx[in0]+1):
            if i > 0:
                li[i][in0] = 1
        flag = 0

    return li


def pixel_label_mkr(img, bbox_label, thresh, cls, instance_seg):
    cls_cnt = 0
    sorted_list = list_sort(bbox_label)
    pixel_label = np.zeros((w, h), dtype=np.uint8)
    for i in sorted_list:
        # pixel_label[i[1]:i[3], i[0]:i[2]] = (img[i[1]:i[3], i[0]:i[2]] > thresh)*cls
        raw_arr = (img[i[1]:i[3], i[0]:i[2]] > thresh)*(cls + (cls_cnt*instance_seg))
        li = fill_arr(raw_arr)
        pixel_label[i[1]:i[3], i[0]:i[2]] = li
        cls_cnt += 1

    if SAVE_PIXEL_LABEL == 1:
        save_path = f"../{save_dir}/PIXEL_LABEL"
        img_name = f"{df.iloc[IDX,0]}.png"
        if not os.path.exists(save_path):  os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/{img_name}", pixel_label)
    return pixel_label

## -------------------------------------------------------------------------------- IMAGE FUNCTION
def convert_to_format():
    global df
    df['count'] = df['count'].fillna(0)
    df = pd.read_csv(file)
    df1 = df.replace({'.png': '.jpg'}, regex=True)
    df1 = df1.rename(columns={'img_name': 'rgb_name'})
    df = df.drop(df.columns[1], axis=1)
    df = pd.concat([df, df1], axis=1)
    df["count"] = ""


def drop_err(csv):
    img_list = glob(f"{save_dir}/*.png")
    df = pd.read_csv(save_csv_path)
    for IDX in range(len(df)):
        if f"out/{df.iloc[IDX, 0]}" not in img_list:
            df.iloc[IDX] = np.nan
    df = df.dropna()
    df.iloc[:, 1] = df.iloc[:, 1].astype(np.uint8)
    df.to_csv("output(err_dropped).csv", index=False)


def histogramer(img):
    im = img.flatten()
    plt.hist(im, bins=range(img.min(), img.max(), 1), histtype='bar', edgecolor='yellow', color='green')
    plt.show()


# def crop_img(arr, W, H, X, Y, a, b, R, case=0):
#     new_arr = arr
#     if case == 1:
#         for x in range(W):
#             for y in range(H):
#                 if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > R**2:
#                     new_arr[y, x] = 0
#     new_arr = new_arr[:, W//5:W-W//5]
#     new_arr = new_arr[H//4:, :]
#
#     return new_arr


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

    # high = 255
    # img *= high//img.max()
    # img[img > high] = high

    # img = img1 * img
    # img[abs(img) < thresh] = img.min()
    # histogramer(img)

    # bg_img = cv2.cvtColor(bg_img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    # bg_img = clahe.apply(bg_img)

    return img.astype(np.uint8)


def bg_mkr(img):
    global BG_LIST, bbox_list, ERROR
    error1 = len(img[img > 237])
    error2 = len(img[img < 1])
    if error1 > 512 or error2 > 256:
        log(f"{IDX}: white-{error1}, black-{error2}")
        bbox_list = [0]
        ERROR = 1
        insert_to_df()
        bg = np.zeros([h, w], dtype=int)
        return 0, bg
    else:
        # log(f"{IDX}: white-{error1}, black-{error2}")
        BG_LIST.append(img)
        if len(BG_LIST) > BG_LENGTH:  BG_LIST.pop(0)
        else:  pass
        bg = bg_filter(img)
        return 1, bg


def open_new_img():
    global base_name, df, image1, image2

    image1 = cv2.imread(rgb, 1)[:, :, ::-1]
    image2 = cv2.imread(ir, 0)

    try:
        if BG_RM == 1:
            num, bg = bg_mkr(image2)
            if BG == 1:  image1 = bg
            elif BG == 2:  image2 = bg
        if SAVE_IMG == 1:
            base_name = os.path.basename(ir)
            save_img_path = f"{save_dir}/{base_name}"
            cv2.imwrite(save_img_path, bg)
        # crop_image(image1, 16, 38, 40, side=-1)
        # crop_image(image2, 3.2, 7.6, 8, side=1)
        resize_image(image1, side=-1)
        resize_image(image2, side=1)

    except TypeError as TE:
        print(TE)


def crop_image(img, a, b, R, side):
    global cropped_image

    cropped_image = img
    H, W = img.shape[0], img.shape[1]
    X, Y = H//2, W

    if side == 1 or side == 2:
        for x in range(W):
            for y in range(H):
                if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > R**2:
                    cropped_image[y, x] = 0
        cropped_image = cropped_image[:, W//5:W-W//5]
        cropped_image = cropped_image[H//4:, :]
        if side == 2:
            return cropped_image
    elif side == -1:
        cropped_image = cropped_image[X1:X2, Y1:Y2]

    resize_image(cropped_image, side)


def resize_image(img, side):
    global resized_image1, resized_image2

    resized_image = cv2.resize(img, re_size, interpolation=cv2.INTER_NEAREST)  ## INTER_NEAREST / CUBIC / LINEAR / AREA

    if side == -1:
        resized_image1 = resized_image.copy()
        param2 = len(bbox_list)
    if side == 0:
        resized_image1 = resized_image.copy()
        param2 = new_label
    elif side == 1:
        resized_image2 = resized_image.copy()
        param2 = IDX
    draw_txt(resized_image, param2, side)


def draw_txt(img, label, side):
    global new_cnt_img

    if DOT == 1:
        for l in dot_list:
            img = cv2.circle(img, center=(l[1]*SIZE, l[0]*SIZE), radius=1, color=(127, 0, 0), thickness=1, lineType=1)

    if len(bbox_list) > 0:
        if bbox_list[0] == -1 or bbox_list[0] == 0 or bbox_list[0] == [0, 0, 0, 0]:
            pplcnt = 0
        else:
            pplcnt = label
    else:
        pplcnt = 0

    if side < 0:
        x, y = img.shape[0]-a1, b1
        pplcnt = pplcnt
    elif side > 0:
        x, y = a2, b2
        pplcnt = IDX
    new_cnt_img = cv2.putText(img, str(pplcnt), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 255), thickness=5, lineType=1)

    draw_rec(new_cnt_img, side)


def draw_rec(img, side=0):
    global bbox_list

    new_img = img
    if len(bbox_list) > 0:
        if bbox_list[0] == -1 or bbox_list[0] == 0 or bbox_list[0] == [0, 0, 0, 0]:
            pass
        else:
            if BBOX == 1:
                for l in bbox_list:
                    try:
                        new_img = cv2.rectangle(new_img, (l[0]*SIZE, l[1]*SIZE), (l[2]*SIZE, l[3]*SIZE), color=(255, 0, 0), thickness=1, lineType=1)
                    except Exception as E:
                        print(f'[!!!] {E}{chr(10)}{traceback.format_exc()}')
                        pass
            if PIXEL == 1:
                if side > 0:
                    pixel_label = pixel_label_mkr(image2, bbox_list, pixel_thresh, 1, INSTANCE_SEG)
                    pixel_label = pixel_label*255
                    pixel_label = cv2.resize(pixel_label, re_size, interpolation=cv2.INTER_NEAREST)
                    if DILATE == 1:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  ## RECT  ELLIPSE  CROSS
                        # pixel_label = cv2.morphologyEx(pixel_label, cv2.MORPH_CLOSE, kernel)
                        pixel_label = cv2.dilate(pixel_label, kernel, iterations=2)
                    alpha = 0.5
                    new_img = cv2.addWeighted(new_img, alpha, pixel_label, 1-alpha, 0)
    else:
        pass
    show_image(new_img, side)


def show_image(img, side):
    global image
    image = ImageTk.PhotoImage(Image.fromarray(img))
    if AUTO_RUN == 0:
        if side < 0:
            LEFT_IMG.configure(image=image)
            LEFT_IMG.image = image
        if side > 0:
            RIGHT_IMG.create_image(0, 0, image=image, anchor='nw')
            RIGHT_IMG.image = image

## -------------------------------------------------------------------------------- EVENT FUNCTION
def log(info):
    if DEBUG == 1:
        print(info)

def clear():
    count_text.set(0)
    index.delete(0, END)

def change_count(cnt):
    global new_label
    new_label = count_text.get()+cnt
    count_text.set(new_label)

def get_mouse_xy(e):
    x, y = e.x, e.y
    log(f"{x}, {y}")
    return x, y

def insert_to_df():
    global bbox_list, ERROR, df1
    if ERROR == 0:
        bbox_len = len(bbox_list)
        if bbox_len == 0:
            # bbox_list = []
            bbox_list = [[0, 0, 0, 0]]
        elif bbox_len > 1:
            if bbox_list[0] == [0, 0, 0, 0] or bbox_list[0] == 0 or bbox_list[0] == -1:
                bbox_list.pop(0)
        json_loc = json.dumps(bbox_list)
        df.iloc[IDX, 1] = json_loc
        # new_row = {'path': [df.iloc[IDX, 0]], 'label': [json_loc]}
        # df2 = pd.DataFrame(new_row)
        # df1 = pd.concat([df1,df2], ignore_index=True)
        # df1.reset_index()
    ERROR = 0

def read_data(n):
    global IDX, ir, rgb, data, bbox_list, BBOX_CNT
    clear()
    IDX += n
    data = df.iloc[IDX, 0]
    try:
        bbox_list = json.loads(df.iloc[IDX, 1])
    except Exception as E:
        bbox_list = []
        log(f"[!!] {E}: {IDX}")
        traceback.print_exc()
        pass
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
    except Exception as E:
        log(f"[!!] {E}: {IDX}")
        traceback.print_exc()

    if AUTO_RUN == 0:
        img = resized_image2.copy()
        draw_txt(img, IDX, side=1)

## -------------------------------------------------------------------------------- EVENT
def select_directoru():
    global data_dir
    if DEBUG == 0:
        data_dir = filedialog.askdirectory()
        data_path.delete(0, END)
    data_path.insert(0, data_dir)

def select_csv():
    global df, data_dir
    file = filedialog.askopenfile()
    root.title(f"{file.name}")
    df = pd.read_csv(file)
    df.sort_values(by=df.keys()[0], inplace=True, ascending=True)
    data_dir = data_path.get()
    read_data(0)

def select_label(e):
    global LABEL_TYPE
    LABEL_TYPE = selected_label.get()
    if LABEL_TYPE == "bbox":
        print(f'{LABEL_TYPE}')
    elif LABEL_TYPE == "pixel":
        print(f'{LABEL_TYPE}')
    elif LABEL_TYPE == "count":
        print(f'{LABEL_TYPE}')

def clone_bbox(e):
    global bbox_list
    bbox_list = json.loads(df.iloc[IDX-1, 1])

    img1 = resized_image1.copy()
    draw_txt(img1, len(bbox_list), side=-1)
    img2 = resized_image2.copy()
    draw_txt(img2, IDX, side=1)

def delete_bbox(e):
    x, y = get_mouse_xy(e)
    f1 = 0
    x, y = round(x/SIZE), round(y/SIZE)
    for idx, i in enumerate(bbox_list):
        if i[0] < x < i[2] and i[1] < y < i[3]:
            bbox_list.pop(idx)
            f1 = 1
            break
    try:
        if f1 == 0:  bbox_list.pop(-1)
    except Exception as E:
        bbox_list.append([0, 0, 0, 0])
        print(f'{E}{chr(10)}{traceback.format_exc()}')
        pass

    img1 = resized_image1.copy()
    draw_txt(img1, len(bbox_list), side=-1)
    img2 = resized_image2.copy()
    draw_txt(img2, IDX, side=1)

def draw_bbox(e):
    global bboxId

    if STATE['click'] == 0:
        STATE['x'], STATE['y'] = e.x, e.y
    else:
        x1, x2 = round(min(STATE['x'], e.x)/SIZE), round(max(STATE['x'], e.x)/SIZE)
        y1, y2 = round(min(STATE['y'], e.y)/SIZE), round(max(STATE['y'], e.y)/SIZE)
        for i, j  in enumerate(bbox_list):
            if j == [0, 0, 0, 0]:
                bbox_list.pop(i)
        bbox_list.append([x1, y1, x2, y2])
        # bboxIdList.append(bboxId)
        # bboxId = None
        # listbox.insert(END, '%s : (%d, %d) -> (%d, %d)'%(currentLabelclass, x1, y1, x2, y2))
        # listbox.itemconfig(len(bbox_list)-1, fg=COLORS[(len(bbox_list)-1)%len(COLORS)])
    STATE['click'] = 1-STATE['click']

    img1 = resized_image1.copy()
    draw_txt(img1, len(bbox_list), side=-1)
    img2 = resized_image2.copy()
    draw_txt(img2, IDX, side=1)

def show_mouse(e):
    global hl, vl, bboxId
    # disp.config(text='x: %d, y: %d'%(e.x, e.y))
    if image:
        if hl:
            RIGHT_IMG.delete(hl)
        hl = RIGHT_IMG.create_line(0, e.y, image.width(), e.y, fill='white', width=1)
        if vl:
            RIGHT_IMG.delete(vl)
        vl = RIGHT_IMG.create_line(e.x, 0, e.x, image.height(), fill='white', width=1)
    if STATE['click'] == 1:
        if bboxId:
            RIGHT_IMG.delete(bboxId)
        COLOR_INDEX = len(bboxIdList)%len(COLORS)
        bboxId = RIGHT_IMG.create_rectangle(STATE['x'], STATE['y'], e.x, e.y,
                                            width=1, outline=COLORS[len(bbox_list)%len(COLORS)])

def check_box():
    global DOT, PIXEL, BBOX, BG_RM, DILATE
    b0 = dot_val.get()
    b1 = pixel_val.get()
    b2 = bbox_val.get()
    b3 = bg_val.get()
    b4 = dilate_val.get()
    if b0 == 1:  DOT = 1
    elif b0 == 0: DOT = 0
    if b1 == 1:  PIXEL = 1
    elif b1 == 0: PIXEL = 0
    if b2 == 1:  BBOX = 1
    elif b2 == 0: BBOX = 0
    if b3 == 1:  BG_RM = 1
    elif b3 == 0: BG_RM = 0
    if b4 == 1:  DILATE = 1
    elif b4 == 0: DILATE = 0
    print([b0, b1, b2, b3])
    change(0)

def save_img():
    cv2.imwrite(f'./{data}_{IDX}.png', image2)
    messagebox.showinfo("Information", "img saved succesfully")

def save():
    df.to_csv(f"label_{IDX}.csv", index=False)
    # df1.to_csv(f"new_label_{IDX}.csv", index=False)
    messagebox.showinfo("Information", "saved succesfully")

def move(e):
    global IDX
    IDX = int(index.get())
    change(0)

def loop(e):
    while 1:
        insert_to_df()
        change(1)

def next(e):
    insert_to_df()
    change(1)

def prev(e):
    insert_to_df()
    change(-1)

def close(e):
    df.to_csv(f"backup.csv", index=False)
    root.destroy()

## -------------------------------------------------------------------------------- IMAGE SHIFT

'''
up: X1 +    down: X2 -    left: Y1 +    right: Y2 -    u1: X1 -    d1: X2 +    l1: Y1 -    r1: Y2 +
'''
def shift_image():
    global cropped_image1, resized_image1
    # img = cv2.imread(rgb, 1)[:, :, ::-1]
    img = image1
    print(f'X1: {X1},  X2: {X2},  Y1: {Y1},  Y2: {Y2}')
    print(f'X1, X2, Y1, Y2 = {X1}, {X2}, {Y1}, {Y2}')
    img = img[X1:X2, Y1:Y2]
    resized_image = cv2.resize(img, (960, 960), interpolation=cv2.INTER_NEAREST)
    resized_image1 = resized_image.copy()
    cropped_image1 = img.copy()
    draw_txt(img, len(bbox_list), -1)
    # draw_txt(cropped_image1, len(bbox_list), -1)

def up(e):
    global X1
    X1 += 1
    print(f"X1 +")
    if X1 > s1/2: X1 = s1//2
    shift_image()

def down(e):
    global X2
    X2 -= 1
    print(f"X2 -")
    if X2 < s1/4*3: X2 = s1//4*3
    shift_image()

def left(e):
    global Y1
    Y1 += 1
    print(f"Y1 +")
    if Y1 > s2/2: Y1 = s2//2
    shift_image()

def right(e):
    global Y2
    print(f"Y2 -")
    Y2 -= 1
    if Y2 < s2/4*3: Y2 = s2//4*3
    shift_image()

def up1(e):
    global X1
    X1 -= 1
    print(f"X1 -")
    if X1 < s1_mn: X1 = s1_mn
    shift_image()

def down1(e):
    global X2
    X2 += 1
    print(f"X2 +")
    if X2 > s1_mx: X2 = s1_mx
    shift_image()

def left1(e):
    global Y1
    print(f"Y1 -")
    if Y1 < s2_mn: Y1 = s2_mn
    Y1 -= 1
    shift_image()

def right1(e):
    global Y2
    Y2 += 1
    print(f"Y2 +")
    if Y2 > s2_mx: Y2 = s2_mx
    shift_image()

## -------------------------------------------------------------------------------- WINDOW
# class ResizingCanvas(Canvas):
#     def __init__(s, parent, **kwargs):
#         Canvas.__init__(s, parent, **kwargs)
#         s.bind("<Configure>", s.on_resize)
#         s.height = s.winfo_reqheight()
#         s.width = s.winfo_reqwidth()
#
#     def on_resize(s, event):
#         # determine the ratio of old width/height to new width/height
#         wscale = float(event.width)/s.width
#         hscale = float(event.height)/s.height
#         s.width = event.width
#         s.height = event.height
#         # resize the canvas
#         s.config(width=s.width, height=s.height)
#         # rescale all the objects tagged with the "all" tag
#         s.scale("all", 0, 0, wscale, hscale)


# def main():
    # root = Tk()
    # root_frame = Frame(root)
    # root_frame.pack(fill=BOTH, expand=YES)
    # root_canvas = ResizingCanvas(root_frame, width=850, height=400, bg="red", highlightthickness=0)
    # root_canvas.pack(fill=BOTH, expand=YES)
    #
    # # add some widgets to the canvas
    # root_canvas.create_line(0, 0, 200, 100)
    # root_canvas.create_line(0, 100, 200, 0, fill="red", dash=(4, 4))
    # root_canvas.create_rectangle(50, 25, 150, 75, fill="blue")
    #
    # # tag all of the drawn widgets
    # root_canvas.addtag_all("all")
    # # root.mainloop()

fg_color = '#ffffff'
bg_color = '#313131'
LABEL_TYPE = 'bbox'

root = Tk()
root.title(f"labeler")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
root.configure(bg=bg_color)

## -------------------------------------------------------------------------------- GRID
count_text = StringVar()
count_text.set(0)

## ---------------------------------------------------------------- 0,0
image1_text = StringVar()
image1_text.set('')
label0 = Label(root, textvariable=image1_text, fg=fg_color, bg=bg_color)
label0.grid(row=0, column=0)

## ---------------------------------------------------------------- 0,1
image2_text = StringVar()
image2_text.set('')
label1 = Label(root, textvariable=image2_text, fg=fg_color, bg=bg_color)
label1.grid(row=0, column=1)

## ---------------------------------------------------------------- 0,2



## ---------------------------------------------------------------- 1,0
LEFT_IMG = Label(root, bg=bg_color)
LEFT_IMG.grid(row=1, column=0)

## ---------------------------------------------------------------- 1,1
RIGHT_IMG = Canvas(root, width=re_size[0], height=re_size[1], cursor='tcross', bg=bg_color)
RIGHT_IMG.grid(row=1, column=1)

## ---------------------------------------------------------------- 2,0
index_frame = Frame(root, width=0, height=0, bg=bg_color)
index_frame.grid(row=2, column=0)

index_label = Label(index_frame, text='DATA INDEX :', fg=fg_color, bg=bg_color)
index_label.grid(row=0, column=0, padx=8, pady=8)

index = Entry(index_frame, width=10, justify='center', borderwidth=3, bg='yellow')
index.grid(row=0, column=1, padx=8)

## ---------------------------------------------------------------- 2,1
check_frame = Frame(root, width=0, height=0, bg=bg_color)
check_frame.grid(row=2, column=1, pady=8)

dot_val = IntVar(value=1)
dot_button = Checkbutton(check_frame, text="DOT", variable=dot_val, command=check_box, onvalue=1, offvalue=0, height=1, width=4)
dot_button.grid(row=0, column=0, padx=4)

pixel_val = IntVar(value=1)
pixel_button = Checkbutton(check_frame, text="PIXEL", variable=pixel_val, command=check_box, onvalue=1, offvalue=0, height=1, width=4)
pixel_button.grid(row=0, column=1, padx=4)

bbox_val = IntVar(value=1)
bbox_button = Checkbutton(check_frame, text="BBOX", variable=bbox_val, command=check_box, onvalue=1, offvalue=0, height=1, width=4)
bbox_button.grid(row=0, column=2, padx=4)

bg_val = IntVar(value=1)
raw_button = Checkbutton(check_frame, text="BG", variable=bg_val, command=check_box, onvalue=1, offvalue=0, height=1, width=4)
raw_button.grid(row=0, column=3, padx=4)

dilate_val = IntVar(value=1)
dilate_button = Checkbutton(check_frame, text="DILATE", variable=dilate_val, command=check_box, onvalue=1, offvalue=0, height=1, width=4)
dilate_button.grid(row=0, column=4, padx=4)

## ---------------------------------------------------------------- 3.0
data_path_frame = Frame(root, width=0, height=0, bg=bg_color)
data_path_frame.grid(row=3, column=0, pady=8)

data_0 = Label(data_path_frame, text='DATA PATH :', fg=fg_color, bg=bg_color)
data_0.grid(row=0, column=0, padx=8)

data_path = Entry(data_path_frame, width=64, justify='center', borderwidth=3, bg='white')
data_path.grid(row=0, column=1, padx=8)

## ---------------------------------------------------------------- 3,1
select_frame = Frame(root, width=0, height=0, pady=16, bg=bg_color)
select_frame.grid(row=3, column=1, pady=8)

select_directory_button = Button(select_frame, text="select DIRECTORY", command=select_directoru)
select_directory_button.grid(row=0, column=0, padx=8)

select_csv_button = Button(select_frame, text="select CSV", command=select_csv)
select_csv_button.grid(row=0, column=1, padx=8)

selected_label = StringVar()
selected_label.set("pixel")
labels = ["pixel", "bbox", "count", ]
select_label_drop = OptionMenu(select_frame, selected_label, *labels, command=select_label)
select_label_drop.grid(row=0, column=2, padx=8)

## ---------------------------------------------------------------- 4,0
key_guide_string = "LEFT ARROW: previous\n" \
                   "RIGHT ARROW: next\n" \
                   "LABEL: right click\n" \
                   "DELETE: left click inside box\n" \
                   "DELETE LATEST: left click outside box\n" \
                   "GOTO: write image # on yellow box and press enter"
key_guide_text = StringVar()
key_guide_text.set(key_guide_string)
key_guide = Label(root, textvariable=key_guide_text, fg=fg_color, bg=bg_color)
key_guide.grid(row=4, column=0)

## ---------------------------------------------------------------- 4,1
save_frame = Frame(root, width=0, height=0, pady=16, bg=bg_color)
save_frame.grid(row=4, column=1)

save_img = Button(save_frame, text="save_img", bg='green', fg='purple', padx=10, pady=10, command=save_img)
save_img.grid(row=0, column=1, padx=8)

save = Button(save_frame, text="save_CSV", bg='red', fg='blue', padx=10, pady=10, command=save)
save.grid(row=0, column=2, padx=8)

## -------------------------------------------------------------------------------- BIND
root.bind('<Return>', move)

root.bind('<Right>', next)
root.bind('<Left>', prev)

root.bind('<Escape>', close)

if LABEL_TYPE == 'bbox':
    RIGHT_IMG.bind("<Button 1>", draw_bbox)
elif LABEL_TYPE == 'pixel':
    RIGHT_IMG.bind("<Button 1>", draw_bbox)

RIGHT_IMG.bind('<Button 2>', clone_bbox)
RIGHT_IMG.bind('<Button 3>', delete_bbox)
RIGHT_IMG.bind("<Motion>", show_mouse)

root.bind('<Control-Up>', up)
root.bind('<Control-Down>', down)
root.bind('<Control-Right>', right)
root.bind('<Control-Left>', left)

root.bind('<Shift-Up>', up1)
root.bind('<Shift-Down>', down1)
root.bind('<Shift-Right>', right1)
root.bind('<Shift-Left>', left1)

root.bind('<Control-Shift-Button 2>', loop)

## -------------------------------------------------------------------------------- START
root.mainloop()


# if __name__ == "__main__":
#     main()