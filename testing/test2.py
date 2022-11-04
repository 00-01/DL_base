# import cv2
# import numpy as np
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

a = 10

b = a
print(a)

b -= 1
print(a)
