import json

import numpy as np
import pandas as pd
from PIL import Image


def v2_1_data_mkr(df, z_value):
    cnt0 = 0
    path = []
    label = []
    for i in range(len(df.index)):
        bbox = json.loads(df.iloc[i, 1])
        lbl = len(bbox)
        if lbl == 1:
            if 0 not in bbox:
                if [0, 0, 0, 0] in bbox:
                    ## REMOVE 01_DATA IMBALANCE
                    cnt0 += 1
                    if cnt0%z_value == 0:
                        label.append(0)
                        path.append(df.iloc[i, 0])
                        cnt0 = 0
                else:
                    label.append(lbl)
                    path.append(df.iloc[i, 0])
        elif lbl == 0:
            pass
        else:
            label.append(lbl)
            path.append(df.iloc[i, 0])
    path = np.array(path)
    label = np.array(label)

    df1 = pd.DataFrame(list(zip(path, label)), columns=['path', 'label'])

    return df1


def v2_2_data_mkr(df, z_value):
    cnt0 = 0
    path = []
    label = []
    for i in range(len(df.index)):
        tmp = []
        for j in json.loads(df.iloc[i, 1]):
            try:
                if len(j) == 4:
                    x = j[2]-((j[2]-j[0])/2)
                    y = j[3]-((j[3]-j[1])/2)
                    tmp.append([x, y])
            except Exception as E:
                pass
        if len(tmp) > 0:
            if 0 in tmp[0]:
                cnt0 += 1
                if cnt0%z_value == 0:
                    path.append(df.iloc[i, 0])
                    label.append(tmp)
                    cnt0 = 0
            else:
                path.append(df.iloc[i, 0])
                label.append(tmp)
    path = np.array(path)
    label = np.array(label)

    df1 = pd.DataFrame(list(zip(path, label)), columns=['path', 'label'])

    return df1


def df_to_tensor(df, path):
    data = []
    label = []
    for i in range(len(df)):
        try:
            img = Image.open(f"{path}/{df.iloc[i, 0]}.png")
            data.append(list(img.getdata()))
            label.append(df.iloc[i, 1])
        except Exception as E:
            print(E)
        if i%10000 == 0:  print(i)

    return data, label
