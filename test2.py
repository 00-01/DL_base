import csv
from datetime import datetime
from glob import glob
import json
import os

import cv2
from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Reshape
from matplotlib import pyplot as plt
from numpy import interp
import numpy as np
import pandas as pd
from PIL import Image
# import plotly.express as px
import tensorflow as tf
from tensorflow.python.keras import losses, Model, optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.utils import losses_utils

from helper.eval_function import log, draw_CM, draw_ROC_AUC
from helper.gpu_memory import set_gpu_memmory
from helper.model import con, res, res1, res2

# load tensorboard
%load_ext tensorboard

# %%
"""
0 CPU
1 필요한 만큼 메모리를 런타임에 할당
2 GPU에 할당되는 전체 메모리 크기를 제한
"""
set_gpu_memmory(2)
# %% md
# 0. SET-UP
# %%
DEBUG = 1
TENSORBOARD = 1
SAVE = 1

MIN, MAX = 0, 255

# %% md
# 1. DATASET
# %%
# data_dir = f"/media/z/0/MVPC10/DATA/v1.1/RAW/03"
# file = f"label_1248_cnt.csv"
# file = f"/media/z/0/MVPC10/CODE/pplcnt_model/LABEL/labeled_refined.csv"
file = f"~/LABELING/FINAL_REFINED.csv"
df = pd.read_csv(file)
# df.sort_values(by=df.keys()[0], inplace=True, ascending=True)
df.head

# %%
len(df)
# %%
df.info
# %%
df.empty
# %%
df.dtypes
# %% md
### remoce all exceptiona
# %%
# df[df.iloc[:,1] < 1]
# df.iloc[:,1]
# print(np.where(df[0] > 0))
# %% md
### LABEL
# %%
S1, S2 = 80, 80


def dot_mkr(s1, s2, x, y):
    li = []
    for i in range(x//2, s1, x):
        for j in range(y//2, s2, y):
            li.append([i, j])
    return li


dot_list = dot_mkr(S1, S2, 4, 4)
print(f"len dot_list: {len(dot_list)}")
# print(dot_list)

with open('dot_44.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['x', 'y'])
    write.writerows(dot_list)


# %%
def bbox2dot(dataframe):
    path = []
    label = []
    for i in range(len(dataframe.index)):
        tmp = []
        for j in json.loads(dataframe.iloc[i, 1]):
            try:
                if len(j) == 4:
                    x = j[2]-((j[2]-j[0])/2)
                    y = j[3]-((j[3]-j[1])/2)
                    tmp.append([x, y])
            except Exception as E:
                pass
        if len(tmp) > 0:
            path.append(dataframe.iloc[i, 0])
            label.append(tmp)
    path = np.array(path)
    label = np.array(label)

    df = pd.DataFrame(list(zip(path, label)), columns=['path', 'label'])

    return df


# %%
dot_df = bbox2dot(df)
dot_df.head


# %%
# for i in range(len(dot_df)):
#     if len(dot_df.iloc[i,1]) == 0:
#         print(0)
# %%
def label_mkr(data):
    """in: raw label, out: padded label"""
    label = []
    for r in data:
        li = []
        for d in dot_list:
            dis1 = abs(r[0]-d[0])
            dis2 = abs(r[1]-d[1])
            li.append(dis1+dis2)
        # mn = li.index(min(li))
        dis, idx = li[0], 0
        for i in range(0, len(li)):
            if li[i] < dis:
                dis = li[i]
                idx = i
        label.append(idx)

    return label


# %%
label_li = []
for i in range(0, len(dot_df)):
    label = label_mkr(dot_df.iloc[i, 1])
    label_li.append(label)
# %%
# label_li[3000:3600]
# %%
label_li[0]
# %%
label_padding = np.zeros([len(dot_df), 400, 1], dtype=int)
label_padding.shape
# %%
for idx, ele in enumerate(label_li):
    for e in ele:
        label_padding[idx, e] = 1

label_padding.shape
# %% md
### post process label
# %%
label_padding[78900, 0]
# %%
## remove all ones in first place
for i in range(len(label_padding)):
    label_padding[i, 0] = 0
# %%
label_padding[78900, 0]

# %%
sh1 = int(np.sqrt(label_padding.shape[1]))
label1 = np.reshape(label_padding, (label_padding.shape[0], sh1, sh1, 1))
# label1 = np.reshape(label_padding, (label_padding.shape[0], sh1, sh1))

label1.shape
# label1[36]
# %%
ind = 36

print(label_li[ind])
print(np.where(label_padding[ind] > 0))
# %% md
### DATA
# %%
df1 = dot_df.iloc[:, 0]
df1.head
# %%
# PATH TO REAL_PATH
img_dir = f"/media/z/0/MVPC10/CODE/pplcnt_model/labeling_tool/out"
for i in range(len(df1)):
    df1.iloc[i] = f"{img_dir}/{df1.iloc[i]}.png"
df1.head

# %%

## GET H,W
sample_img = Image.open(df1.iloc[16])
img_array = np.array(sample_img, int)
H, W = img_array.shape

H, W
# %%
## SHUFFLE
# df = df.sample(frac=1).reset_index(drop=True)
# df
# %%
## DROP ERROR
# df1 = df[df.iloc[:, 2] > 0]
# df1

# %% md
### PRE-PROCESS
# %%
# label_padding
# %%
## DATASET TO TENSOR
data = []
for i in range(len(dot_df)):
    try:
        img = Image.open(df1[i])
        img = data.append(list(img.getdata()))
    except Exception as E:
        log(DEBUG, E)
    if i%10000 == 0:  print(i)

data = np.array(data)
data = data.reshape(data.shape[0], H, W, 1)

log(DEBUG, data.shape)
log(DEBUG, label1.shape)

# %%
# label1[36] > 0

# %%
# ## Shuffle
# seed = 99
# np.random.seed(seed)
# np.random.shuffle(data)
# # np.random.seed(seed)
# np.random.shuffle(label)


## Nomalize
# log(DEBUG, data[0][0])
norm_data = data/MAX
# norm_data = data.astype("float")/MAX
# log(DEBUG, norm_data[0][0])

# label = label1
label = label_padding
label = np.reshape(label, (label_padding.shape[0], label_padding.shape[1]))

## TEST SPLIT
split1 = int(len(label)*0.9)
X1, X2 = norm_data[:split1], norm_data[split1:]
Y1, Y2 = label[:split1], label[split1:]
## VAL SPLIT
split2 = int(len(label)*0.9)
# train_data, val_data = train_data[:split2], train_data[split2:]
# train_label, val_label = train_label[:split2], train_label[split2:]

# %% md
# 2. TRAIN
# %%
print(X1.shape)
print(Y1.shape)
# %%
# input = Input(shape=(H, W, 1))
#
# c0 = Conv2D(512, (3, 3), padding='same')(input)
# c0 = BatchNormalization()(c0)
# c0 = Activation('selu')(c0)
#
# # x = res(c0, 512)
# # x = res(x, 256)
# # x = res(x, 128)
# # x = res(x, 64)
# # x = res(x, 32)
# # x = res(x, 16)
#
# x = res1(c0, 64)
# x = res1(x, 64)
# x = res2(x, 64)
# x = res1(x, 64)
# x = res2(x, 128)
# x = res1(x, 128)
# x = res2(x, 256)
# x = res1(x, 256)
# # x = res2(x, 512)
# # x = res1(x, 512)
#
# x = tf.reduce_mean(x, (1,2))#, axis=None, keepdims=False, name=None)
# x = Dropout(.50)(x)
#
# # x = Dense(256, activation="selu")(x)
# # x = Dense(128, activation="selu")(x)
# # x = Dense(64, activation="selu")(x)
# # output = Dense(label.shape[1], activation="softmax")(x)
# output = x
#
# model = Model(input, output)
# %%
## ------------------------------------------------ IN
input = Input(shape=(H, W, 1))

## ------------------------------------------------ HEAD
# x = Conv2D(8, (13, 13), strides=2, padding='valid', activation='selu', name="head_c0")(input)
x = Conv2D(16, (11, 11), strides=1, padding='valid', activation='selu', name="head_c1")(input)
# x = Conv2D(1, (3, 3), strides=2, padding='same', activation='selu', name="c1_conv2d")(x)

## ------------------------------------------------ BODY
x = con(32, (9, 9), 1, 'valid', 'c0', x)
x = con(32, (7, 7), 1, 'valid', 'c1', x)
# x = con(64, (5,5), 1, 'valid', 'c2', x)
x = con(64, (3, 3), 1, 'valid', 'c3', x)
# x = con(1, (5,5), 1, 'valid', 'c4', x)
x = con(64, (3, 3), 1, 'valid', 'c5', x)

# x = res1(x, 64)
# x = res1(x, 64)
# x = res1(x, 64)
# x = res1(x, 64)
# x = res1(x, 64)
# x = res1(x, 64)


# x = con(1, (5,5), 1, 'valid', 'c6', x)
x = con(64, (3, 3), 1, 'valid', 'c7', x)
x = con(128, (3, 3), 1, 'valid', 'c8', x)
x = con(128, (3, 3), 1, 'valid', 'c9', x)
x = con(16, (5, 5), 1, 'valid', 'c10', x)
# x = con(8, (3, 3), 1, 'valid', 'c11', x)
# x = con(64, (5, 5), 1, 'valid', 'c12', x)

## ------------------------------------------------ TAIL
x = Flatten()(x)
# x = tf.reduce_mean(x, (1, 2))  #, axis=None, keepdims=False, name=None)
# x = Dropout(.50)(x)

# print(x.shape)


# x = Dropout(.5)(x)

## ------------------------------------------------ OUT
# output = Conv2D(1, (3, 3), strides=2, padding='valid', activation='selu', name="c00")(x)
# output = Reshape((20,20))(x)
output = Dense(400, activation="selu")(x)
# output = x

## ------------------------------------------------ FINAL
model = Model(input, output)
# %% md
### COMPILE
# %%
## ---------------------------------------------------------------- OPTIMIZER
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=0.001,
#         decay_steps=100000,
#         decay_rate=0.96,
#         staircase=True)

# lr_schedule = k.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4,
#                                                decay_steps=1000, )

# optimizer = optimizers.Adam(learning_rate=lr_schedule)
optimizer = optimizers.Adam(learning_rate=0.0001)

## ---------------------------------------------------------------- LOSS
# loss = losses.MeanAbsoluteError()
# loss = losses.BinaryCrossentropy()
# loss = losses.MeanSquaredError()
loss = losses.BinaryFocalCrossentropy(apply_class_balancing=False,
                                      alpha=0.25,
                                      gamma=2.0,
                                      from_logits=False,
                                      label_smoothing=0.0,
                                      axis=-1,
                                      reduction=losses_utils.ReductionV2.AUTO,
                                      name='binary_focal_crossentropy'
                                      )

# def adaptive_loss():
#     pass

## ---------------------------------------------------------------- METRICS
metrics = ['accuracy']
# metrics = [SparseCategoricalAccuracy()]

## ---------------------------------------------------------------- COMPILE
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

# %% md

### START
# %%
BATCH = 32
EPOCH = 128
ES = 2

## fit
log_path = "logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")
es = EarlyStopping(monitor="val_loss", patience=ES, verbose=2, mode='auto')
tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)

history = model.fit(X1, Y1,
                    validation_split=0.1,
                    # validation_data=(val_data, val_label),
                    batch_size=BATCH,
                    epochs=EPOCH,
                    verbose=1,
                    # callbacks=[es],)
                    callbacks=[es, tensorboard_callback], )

# %%
# BATCH = 32
# EPOCH = 30
#
# lb = test_label.reshape(-1)
# has = {}
#
# # np.array(list(set(test_label.reshape(-1)))).astype(np.int64)
# ls_num = list(map(int, list(set(lb))))
# for n in ls_num:
#     has[n] = []
# for ind in range(len(lb)):
#     # ind는 label 인덱스
#     if len(has[lb[ind]]) < 30:
#         has[lb[ind]].append(ind)
#
# x_data = []
# y_data = []
# for key in has.keys():
#     for ind in has[key]:
#         x_data.append(test_data[ind])
#         y_data.append(test_label[ind])
#
# x_data = np.array(x_data).astype(np.float32)
# y_data = np.array(y_data).astype(np.float32)
# train_data = train_data.astype(np.float32)
# train_label = train_label.astype(np.float32)
#
# for i in range(30):
#     print(i)
#     history = model.fit(train_data, train_label,
#                         # validation_split=0.2,
#                         validation_data=(x_data, y_data),
#                         batch_size=BATCH,
#                         epochs=EPOCH,
#                         verbose=1,
#                         # callbacks=[es],)
#                         # callbacks=[es, tensorboard_callback], )
#                         )
#     model.save('asdf/' + str(i) + '.h5')
# %%
## history to DF
hdf = pd.DataFrame(history.history)
hdf.keys()

## plot history
hdf.plot(figsize=(9, 6), grid=1, xlabel="epoch", label="accuracy")
plt.ylim([0, 2])
plt.show()

# %%
Y2[0, 0]
# %%

# df = px.data.gapminder().query("continent=='Oceania'")
# fig = px.line(hdf, x=hdf.index, y=hdf.values, color=hdf.keys())
# fig.show()

# %%
# epo = 10
# model = tf.keras.models.load_model('asdf/' + str(epo) + '.h5')
# test_label.shape
#
# lb = test_label.reshape(-1)
# has = {}
#
# # np.array(list(set(test_label.reshape(-1)))).astype(np.int64)
#
#
# ls_num = list(map(int, list(set(lb))))
# for n in ls_num:
#     has[n] = []
#
# for ind in range(len(lb)):
#
#     # ind는 label 인덱스
#
#     if len(has[lb[ind]]) < 30:
#         has[lb[ind]].append(ind)
#
# ls_num
#
# x_data = []
# y_data = []
#
#
# for key in has.keys():
#     for ind in has[key]:
#         x_data.append(test_data[ind])
#         y_data.append(test_label[ind])
#
# x_data = np.array(x_data)
# y_data = np.array(y_data)
# x_data.shape
# y_data.shape
#
#
# result = np.argmax(model.predict(x_data), -1)
# cont = 0
# for ind in range(len(result)):
#     if result[ind] == y_data.reshape(-1)[ind]:
#         cont +=1
#
#
# cont
# print(len(ls_num))
# print(cont/len(result))


# %%
# for i in range(10):
#     num = i
#     size = 10
#     interpolation = cv2.INTER_NEAREST  ## CUBIC LINEAR AREA
#
#     img = test_data[num].reshape(60,48)
#
#     img = np.around(img*255)
#
#     img = img.astype(np.uint8)
#
#     img = cv2.resize(img, (img.shape[1]*size, img.shape[0]*size), interpolation=interpolation)
#
#     img = Image.fromarray(img)
#
#     print(i)
#     display(img)

# %% md
# 3. EVALUATE
# %%
loss, acc = model.evaluate(X2, Y2, verbose=1)

predict = model.predict(X2)
# %%
# predict_shaped = np.reshape(predict, (predict.shape[0], predict.shape[1]*predict.shape[2]))
# Y2_shaped = np.reshape(Y2, (Y2.shape[0], Y2.shape[1]*Y2.shape[2]))
predict_shaped = np.reshape(predict, (predict.shape[0], predict.shape[1]))
Y2_shaped = np.reshape(Y2, (Y2.shape[0], Y2.shape[1]))

n = 10
print(f"pred:{chr(10)}{predict_shaped[:n]}{chr(10)}")
print(f"label:{chr(10)}{Y2_shaped[:n]}")

# %%
# predict = model.predict(train_data)
#
# draw_CM(train_label, predict)
# predict[0]
# Y2_shaped.shape

# %%
## CM
# draw_CM(Y2, predict)

## ROC, AUC
# x = label_binarize(predicted, classes=CLASS)
# y = label_binarize(Y2, classes=CLASS)
# draw_ROC_AUC(x, y, CLASS)
# %%
indx = 11
print(np.where(Y2[indx] > 0))
print(np.where(predict[indx] > 1.2))
# %%

ind = np.argpartition(predict, -4)[-4:]
print(ind)

top4 = predict[ind]
print(top4)

# sorted_index_array = np.argsort(predict_shaped)
# sorted_array = predict_shaped[sorted_index_array]

# n = 3
# rslt = sorted_array[-n:]
# print(f"{rslt} largest value:")
# %%
predict_shaped1 = predict_shaped
N = 10
# N = len(predicted)
for i in range(600, 680):
    size = 10
    test_img = Image.fromarray((X2[i]*255).reshape(H, W)).convert('L').resize((W*size, H*size))
    display(test_img)

    for j, k in enumerate(predict_shaped1[i]):
        if k < 0.1:
            predict_shaped1[i, j] = 0
    # predict_shaped[i] = [predict_shaped[i,j] = 0 for j,k in enumerate(predict_shaped[i]) if k < 0.1]
    log(DEBUG, f"predicted:{chr(10)}")
    cnt1 = 0
    for line in range(20):
        li = np.round(predict_shaped1[i, cnt1:cnt1+20], 1)
        # li = predict_shaped1[i, cnt1:cnt1+20]
        log(DEBUG, f"{li}")
        cnt1 += 20
    cnt1 = 0
    log(DEBUG, f"{chr(10)}label:{chr(10)}")
    cnt2 = 0
    for line in range(20):
        log(DEBUG, f"{Y2_shaped[i, cnt2:cnt2+20]}")
        cnt2 += 20
    cnt2 = 0

    # log(DEBUG, f"total_difference: {len(abs(predict_shaped[i]-Y2_shaped[i]))}")
    # log(DEBUG, f"difference: {abs(predict_shaped[i]-Y2_shaped[i])}")

# %%
# launch tensorboard @ localhost:6006
if TENSORBOARD == 1:
    %tensorboard--logdir
    logs/--host
    localhost--port
    6006
# %%
if SAVE == 1:
    file_name = "model/mvpc10_"+datetime.now().strftime("%Y%m%d-%H%M%S")
    model_format = ".h5"
    model_name = file_name+model_format
    model.save(model_name)
