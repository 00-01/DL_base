import numpy as np
from PIL import Image


def v2_1_bias_finder(Y, P, start=-0.03, end=0.10, step=0.01):
    n = len(P)
    bias_num = n
    BIAS = 0
    F = 0
    pp_li = []
    for pp in np.arange(start, end, step):
        predict1 = P+pp
        for i in range(n):
            diff = abs(round(predict1[i])-Y[i])
            F += diff
        pp0 = round(pp, 2)
        pp_li.append((pp0, F))
        if bias_num > F:
            bias_num = F
            BIAS = pp0
        F = 0

    for i in pp_li:
        print(i)
    print()
    print(f"BIAS: {BIAS}")

    return BIAS


def v2_1_accuracy_calculator(Y, P, BIAS):
    predict_1 = P+BIAS
    n = len(P)
    wrong = 0
    Y2_cnt = 0
    # F_list = []
    for i in range(n):
        Y2_cnt += Y[i]
        diff = abs(round(predict_1[i])-Y[i])
        wrong += diff
        # F_list.append(diff[0])
    print(f"error: {wrong}")
    print(f"total: {Y2_cnt}")
    print(f"acc:   {round((Y2_cnt-wrong)/Y2_cnt, 2)*100}%")


def v2_1_view_samples1(X, Y, P, thresh=10):
    n = len(Y)
    for i in range(0, n, 1):
        if Y[i] > thresh:
            diff = abs(round(P[i])-Y[i])
            print(f"pred: {P[i]:0.2f}")
            print(f"labl: {Y[i]}")
            print(f"diff: {diff}")
            size = 8
            test_img = Image.fromarray((X[i]*255).reshape(X.shape[1], X.shape[2])).convert('L').resize((X.shape[1]*size, X.shape[2]*size))
            display(test_img)


