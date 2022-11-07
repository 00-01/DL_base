import cv2
import numpy as np
#
#
# ir_path = f'/media/z/0/MVPC10/CODE/DL_base/SAMPLE/data/1651310296797.png'
# img = cv2.imread(ir_path, 0)
#
# arr = [[1, 69, 16, 79], [58, 45, 80, 72], [56, 39, 71, 60], [54, 24, 69, 46],
#        [40, 37, 51, 63], [30, 41, 39, 64], [44, 31, 52, 45], [46, 27, 54, 37],
#        [27, 33, 35, 45], [30, 25, 38, 37], [47, 11, 57, 29], [37, 18, 45, 34],
#        [42, 15, 47, 22], [43, 12, 48, 19], [39, 10, 45, 19], [40, 7, 45, 12],
#        [38, 5, 42, 11], [41, 3, 45, 8], [37, 4, 40, 8], [30, 5, 35, 18], [27, 4, 32, 13]]
#
#
# def list_sort(arr):
#     padded_list = [None]*80
#     for i in arr:
#         try:
#             if i[1] > 0:
#                 if padded_list[i[1]] == None:
#                     padded_list[i[1]] = [i]
#                 else:
#                     padded_list[i[1]].append(i)
#         except Exception as E:
#             pass
#
#     sorted_list = []
#     for i in padded_list:
#         if i != None:
#             if len(i) == 1:
#                 sorted_list.append(i[0])
#             else:
#                 for j in i:
#                     sorted_list.append(j)
#
#     return sorted_list
#
#
# def pixel_label_mkr(img, bbox_label, thresh):
#     sorted_list = list_sort(bbox_label)
#     pixel_label = np.zeros((80, 80), dtype=np.uint8)
#     for i in sorted_list:
#         pixel_label[i[0]:i[2], i[1]:i[3]] = img[i[0]:i[2],i[1]:i[3]] > thresh
#             print(i)
#         # loop all i
#         # img[i[0, 1]]
#     return pixel_label
#
#
# pixel_label = pixel_label_mkr(img, arr, 30)

# li = [[1, 1, 1, 1, 1],
#       [1, 0, 1, 0, 1],
#       [0, 0, 0, 1, 1],
#       [1, 1, 0, 0, 0],
#       [1, 0, 0, 1, 0],
#       [0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0],
#       [0, 0, 0, 0, 0],
#       [0, 0, 1, 0, 0]]

li = [[0, 0, 0, 0, 0],
      [1, 0, 1, 0, 1],
      [0, 0, 0, 1, 1],
      [0, 1, 0, 0, 0],
      [1, 0, 0, 1, 0],
      [0, 0, 0, 0, 0],
      [0, 1, 0, 0, 1],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 1],
      [1, 1, 1, 1, 1]]

for l in li:
    print(l)
print()










# flg0, flg1 = 0, 0
# mn0, mx0 = 0, -1
# mn1, mx1 = 0, -1
# ax0, ax1 = np.shape(li)
# for ind0, val0 in enumerate(li):  ## full-list axis 0
#     print(ind0, val0)
#     if 1 in val0:
#         print(f"flag0: {flg0}")
#         for ind1, val1 in enumerate(val0):  ## full-list axis 1
#             if val1 == 1 and flg0 == 0:
#                 flg0 = 1
#                 flg1 = 1
#                 mn0 = ind1
#                 mn1 = ind1
#                 for a0 in range(ind0, ax0):  ## cross-list axis 0
#                     for a1 in range(ind1, ax1):  ## cross-list axis 1
#                         if a0 == ind0:
#                             if li[a0][a1] == 1:
#                                 mx0 = a1
#                         if a1 == ind1:
#                             if li[a0][a1] == 1:
#                                 mx1 = a0
#                         print(f"{a0}   {a1}   {li[a0][a1]}")
#                         # print(f"{ind0} {ind1}")
#                         # print(f"mn0: {mn0}, mx0: {mx0}")
#                         print(f"mn1: {mn1}, mx1: {mx1}")
#                     print()
#     for m0 in range(mn0, mx0+1):
#         if m0 > 0:
#             li[ind0][m0] = 1
#     # for m1 in range(mn1, mx1+1):
#     #     if m1 > 0:
#     #         li[m1][ind0] = 1
#     #     print(f"m1: {m1}, ind0: {ind0}, {li[m1][ind0]}")
#     flg0, flg1 = 0, 0
#     mn0, mx0 = 0, 0
#     mn1, mx1 = 0, 0
#
#     print()





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

for l in li:
    print(l)
print()
    # for i in range(ax1):  ## full-list axis 0
#     print(i)
    # if 1 in val0:
    #     print(f"flag0: {flg0}")
    #     for ind1, val1 in enumerate(val0):  ## full-list axis 1
    #         if val1 == 1 and flg0 == 0:
    #             flg0 = 1
    #             flg1 = 1
    #             mn0 = ind1
    #             mn1 = ind1
    #             for a0 in range(ind0, ax0):  ## cross-list axis 0
    #                 for a1 in range(ind1, ax1):  ## cross-list axis 1
    #                     if a0 == ind0:
    #                         if li[a0][a1] == 1:
    #                             mx0 = a1
    #                     if a1 == ind1:
    #                         if li[a0][a1] == 1:
    #                             mx1 = a0
    #                     print(f"{a0}   {a1}   {li[a0][a1]}")
    #                     # print(f"{ind0} {ind1}")
    #                     # print(f"mn0: {mn0}, mx0: {mx0}")
    #                     print(f"mn1: {mn1}, mx1: {mx1}")
    #                 print()
    # # for m0 in range(mn0, mx0+1):
    # #     if m0 > 0:
    # #         li[ind0][m0] = 1
    # for m1 in range(mn1, mx1+1):
    #     if m1 > 0:
    #         li[m1][ind0] = 1
    #         print(f"m1: {m1}, ind0: {ind0}, {li[m1][ind0]}")
    # flg0, flg1 = 0, 0
    # mn0, mx0 = 0, 0
    # mn1, mx1 = 0, 0

    # print(mn, mx)




# for i in li:
#     print(i)

li1 = [[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0],
       [1, 1, 1, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]]



# 0 4
# 0 4