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
BG = 2
SAVE_IMG = 1
AUTO_RUN = 0
# DRAW_CNT = 1
MAKE_DENOISE_LABEL = 1

## -------------------------------------------------------------------------------- RUN

IDX, bbox_list, BBOX_CNT = 0, [], 0
h, w = 80, 80
SIZE = 12
# SIZE = 11
re_size = (h*SIZE, w*SIZE)

s1, s2 = 400, 400
ss = h*SIZE/s1
s_size = (s1*ss, s2*ss)
s1_mn, s1_mx, s2_mn, s2_mx = s1//4, s1, s2//5, s2-s2//5
# X1, X2, Y1, Y2 = s1_mn, s1_mx, s2_mn, s2_mx
# X1, X2, Y1, Y2 = 103, 348, 104, 309
X1, X2, Y1, Y2 = 103, 354, 104, 312

error = 239
cnt_thresh = 1
j = 0
new_label = 0

BG_LIST = []
BG_LENGTH = 32

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
save_dir = f"out"
save_csv_path = f"{save_dir}/output.csv"
data_dir = f"/media/z/0/MVPC10/DATA/v1.1/RAW/03"

## -------------------------------------------------------------------------------- DOT
DOT = 1
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
    # counts, bins = np.histogram(im, range=(0,255))
    # plot histogram centered on values 0..255
    # plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    # plt.bar(bins[:-1]-0.5, counts, width=1, edgecolor='none')
    # plt.xlim([-0.5, 255.5])
    plt.hist(im, bins=range(img.min(), img.max(), 1), histtype='bar', edgecolor='yellow', color='green')

    # n, bins, patches = plt.hist(x=im, bins='auto', color='#0504aa', alpha=0.7)
    # plt.grid(axis='both', alpha=0.5)
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # Set a clean upper y-axis limit.
    # maxfreq = n.max()
    # plt.ylim(ymax=np.ceil(maxfreq/10)*10 if maxfreq%10 else maxfreq+10)
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
    bg = np.zeros([h, w], dtype=int)
    for IDX in BG_LIST:
        bg += IDX
    bg //= len(BG_LIST)

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
    global BG_LIST, bbox_list
    error1 = len(img[img > 237])
    error2 = len(img[img < 1])
    if error1 > 512:
        log(f"{IDX}: white-{error1}, black-{error2}")
        bbox_list = [0]
        # bbox_list.append([0,0,0,0])
        insert_to_df()
        bg = np.zeros([h, w], dtype=int)
        return 0, bg
    else:
        # log(f"{IDX}: white-{error1}, black-{error2}")
        BG_LIST.insert(0, img)
        if len(BG_LIST) > BG_LENGTH:  BG_LIST.pop(-1)
        else:  pass
        bg = bg_filter(img)
        return 1, bg


def open_new_img():
    global base_name, df, image1

    image1 = cv2.imread(rgb, 1)[:, :, ::-1]
    image2 = cv2.imread(ir, 0)

    try:
        num, bg = bg_mkr(image2)
        if BG == 1:  image1 = bg
        elif BG == 2:  image2 = bg
        if SAVE_IMG == 1:
            base_name = os.path.basename(ir)
            save_img_path = f"{save_dir}/{base_name}"
            cv2.imwrite(save_img_path, bg)
        crop_image(image1, 16, 38, 40, side=-1)
        crop_image(image2, 3.2, 7.6, 8, side=1)
        return num

    except TypeError as TE:
        print(TE)


def crop_image(img, a, b, R, side):
    cropped_image = img
    H, W = img.shape[0], img.shape[1]
    X, Y = H//2, W

    if side == 1:
        for x in range(W):
            for y in range(H):
                if (((X-x)**2)/a**2)+(((Y-y)**2)/b**2) > R**2:
                    cropped_image[y, x] = 0
        cropped_image = cropped_image[:, W//5:W-W//5]
        cropped_image = cropped_image[H//4:, :]
    elif side == -1:
        cropped_image = cropped_image[X1:X2, Y1:Y2]

    resize_image(cropped_image, side)


def resize_image(img, side):
    global resized_image1, resized_image2

    interpolation = cv2.INTER_NEAREST  ## CUBIC LINEAR AREA
    resized_image = cv2.resize(img, (re_size), interpolation=interpolation)

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

    if side < 0:  x, y = img.shape[0]-a1, b1
    elif side > 0:  x, y = a2, b2

    if DOT == 1:
        for l in dot_list:
            img = cv2.circle(img, center=(l[1]*SIZE, l[0]*SIZE), radius=1, color=(127, 0, 0), thickness=1, lineType=1)

    new_cnt_img = cv2.putText(img, str(label), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(255, 0, 255), thickness=5, lineType=1)

    draw_rec(new_cnt_img, side)

def draw_rec(img, side=1):
    new_img = img
    if len(bbox_list) > 0:
        if bbox_list[0] == -1 or bbox_list[0] == 0 or bbox_list[0] == [0,0,0,0]:
            pass
        else:
            for l in bbox_list:
                try:
                    new_img = cv2.rectangle(new_img, (l[0]*SIZE, l[1]*SIZE), (l[2]*SIZE, l[3]*SIZE), color = (255, 0, 0), thickness = 1, lineType = 1)
                except Exception as e:
                    trace_back = traceback.format_exc()
                    print(f'[!!!] {e}{chr(10)}{trace_back}')
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
    global bbox_list
    bbox_len = len(bbox_list)
    if bbox_len == 0:
        # bbox_list = []
        bbox_list.append([0,0,0,0])
    elif bbox_len > 1:
        if bbox_list[0] == [0,0,0,0] or bbox_list[0] == 0 or bbox_list[0] == -1:
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

## -------------------------------------------------------------------------------- EVENT

def open_folder():
    global data_dir
    data_dir = filedialog.askdirectory()
    data_path.delete(0, END)
    data_path.insert(0, data_dir)

def open_csv():
    global df, data_dir
    file = filedialog.askopenfile()
    win.title(f"{file.name}")
    df = pd.read_csv(file)
    df.sort_values(by=df.keys()[0], inplace=True, ascending=True)
    data_dir = data_path.get()
    read_data(0)

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
        if i[0] < x < i [2] and i[1] < y < i[3]:
            bbox_list.pop(idx)
            f1 = 1
            break
    if f1 == 0:  bbox_list.pop(-1)

    img1 = resized_image1.copy()
    draw_txt(img1, len(bbox_list), side=-1)
    img2 = resized_image2.copy()
    draw_txt(img2, IDX, side=1)

def draw_bbox(e):
    global bboxId

    if STATE['click'] == 0:
        STATE['x'], STATE['y'] = e.x, e.y
    else:
        x1, x2 = round(min(STATE['x'], e.x) / SIZE), round(max(STATE['x'], e.x) / SIZE)
        y1, y2 = round(min(STATE['y'], e.y) / SIZE), round(max(STATE['y'], e.y) / SIZE)
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

def save_img():
    cv2.imwrite(f'./{data}_{IDX}.png', cropped_image2)
    messagebox.showinfo("Information", "img saved succesfully")

def save():
    df.to_csv(f"label_{IDX}.csv", index=False)
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
    win.destroy()

## -------------------------------------------------------------------------------- IMAGE SHIFT

'''
up: X1 +
down: X2 -
left: Y1 +
right: Y2 -

u1: X1 -
d1: X2 +
l1: Y1 -
r1: Y2 +
'''
def shift_image():
    global cropped_image1, resized_image1
    # img = cv2.imread(rgb, 1)[:, :, ::-1]
    img = image1

    print(f'X1: {X1},  X2: {X2},  Y1: {Y1},  Y2: {Y2}')
    print(f'X1, X2, Y1, Y2 = {X1}, {X2}, {Y1}, {Y2}')

    img = img[X1:X2, Y1:Y2]

    resized_image = cv2.resize(img, (960,960), interpolation=cv2.INTER_NEAREST)
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

# def get_x_and_y(e):
#     global lasx, lasy
#     lasx, lasy = e.x, e.y
#
# def draw_smth(e):
#     global lasx, lasy
#     RIGHT_IMG.create_line((lasx, lasy, e.x, e.y), fill='red', width=1)
#     lasx, lasy = e.x, e.y

## -------------------------------------------------------------------------------- WINDOW

win = Tk()
win.title(f"labeler")
win.geometry('1924x1200')
# win.geometry('1900x1000')

## -------------------------------------------------------------------------------- GRID

count_text = StringVar()
count_text.set(0)

## -------------------------------- 0,0
image1_text = StringVar()
image1_text.set('')
label0 = Label(win, textvariable=image1_text)
label0.grid(row=0, column=0)
## -------------------------------- 0,1
image2_text = StringVar()
image2_text.set('')
label1 = Label(win, textvariable=image2_text)
label1.grid(row=0, column=1)

## -------------------------------- 1,0
LEFT_IMG = Label(win)
LEFT_IMG.grid(row=1, column=0)
## -------------------------------- 1,1
RIGHT_IMG = Canvas(win, width=re_size[0], height=re_size[1], cursor='tcross', bg='white')
RIGHT_IMG.grid(row=1, column=1)
RIGHT_IMG.bind("<Button 1>", draw_bbox)
RIGHT_IMG.bind('<Button 2>', clone_bbox)
RIGHT_IMG.bind('<Button 3>', delete_bbox)
RIGHT_IMG.bind("<Motion>", show_mouse)

## -------------------------------- 2,0
index = Entry(win, width=10, justify='center', borderwidth=3, bg='yellow')
index.grid(row=2, column=0)

## -------------------------------- 2,1
select_frame = Frame(win, width=0, height=0, bg='white')
select_frame.grid(row=2, column=1)

open_folder_button = Button(select_frame, text="select folder", command=open_folder)
open_folder_button.grid(row=0, column=1)

open_csv_button = Button(select_frame, text="select csv_file", command=open_csv)
open_csv_button.grid(row=0, column=2)

## -------------------------------- 3.0
data_path_frame = Frame(win, width=0, height=0, bg='white')
data_path_frame.grid(row=3, column=0)

data_0 = Label(data_path_frame, text='DATA PATH: ')
data_0.grid(row=0, column=0)

data_path = Entry(data_path_frame, width=64, justify='center', borderwidth=3, bg='white')
data_path.grid(row=0, column=1)

## -------------------------------- 3,1
save_frame = Frame(win, width=0, height=0, bg='white')
save_frame.grid(row=3, column=1)

save_img = Button(save_frame, text="save_img", bg='green', fg='purple', padx=10, pady=10, command=save_img)
save_img.grid(row=0, column=1)

save = Button(save_frame, text="save_CSV", bg='red', fg='blue', padx=10, pady=10, command=save)
save.grid(row=0, column=2)

## -------------------------------- 4,0
key_guide_string = "LEFT ARROW: previous\n" \
                   "RIGHT ARROW: next\n" \
                   "LABEL: right click\n" \
                   "DELETE: left click inside box\n" \
                   "DELETE LATEST: left click outside box\n" \
                   "GOTO: write image # on yellow box and press enter"
key_guide_text = StringVar()
key_guide_text.set(key_guide_string)
key_guide = Label(win, textvariable=key_guide_text)
key_guide.grid(row=4, column=0)

## -------------------------------------------------------------------------------- BIND

win.bind('<Return>', move)

win.bind('<Right>', next)
win.bind('<Left>', prev)

win.bind('<Escape>', close)

# win.bind('<Button 1>', draw)
# win.bind('<Button 2>', clone)
# win.bind('<Button 3>', erase)

win.bind('<Control-Up>', up)
win.bind('<Control-Down>', down)
win.bind('<Control-Right>', right)
win.bind('<Control-Left>', left)

win.bind('<Shift-Up>', up1)
win.bind('<Shift-Down>', down1)
win.bind('<Shift-Right>', right1)
win.bind('<Shift-Left>', left1)

win.bind('<Control-Shift-Button 2>', loop)

## --------------------------------------------------------------------------------

win.mainloop()

