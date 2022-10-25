# import numpy as np
#
#
# # dot_list = [[2, 2], [2, 6], [2, 10], [2, 14], [2, 18], [2, 22], [2, 26], [2, 30], [2, 34], [2, 38], [2, 42], [2, 46], [2, 50], [2, 54], [2, 58], [2, 62], [2, 66], [2, 70], [2, 74], [2, 78], [6, 2], [6, 6], [6, 10], [6, 14], [6, 18], [6, 22], [6, 26], [6, 30], [6, 34], [6, 38], [6, 42], [6, 46], [6, 50], [6, 54], [6, 58], [6, 62], [6, 66], [6, 70], [6, 74], [6, 78], [10, 2], [10, 6], [10, 10], [10, 14], [10, 18], [10, 22], [10, 26], [10, 30], [10, 34], [10, 38], [10, 42], [10, 46], [10, 50], [10, 54],
# #             [10, 58], [10, 62], [10, 66], [10, 70], [10, 74], [10, 78], [14, 2], [14, 6], [14, 10], [14, 14], [14, 18], [14, 22], [14, 26], [14, 30], [14, 34], [14, 38], [14, 42], [14, 46], [14, 50], [14, 54], [14, 58], [14, 62], [14, 66], [14, 70], [14, 74], [14, 78], [18, 2], [18, 6], [18, 10], [18, 14], [18, 18], [18, 22], [18, 26], [18, 30], [18, 34], [18, 38], [18, 42], [18, 46], [18, 50], [18, 54], [18, 58], [18, 62], [18, 66], [18, 70], [18, 74], [18, 78], [22, 2], [22, 6], [22, 10], [22, 14],
# #             [22, 18], [22, 22], [22, 26], [22, 30], [22, 34], [22, 38], [22, 42], [22, 46], [22, 50], [22, 54], [22, 58], [22, 62], [22, 66], [22, 70], [22, 74], [22, 78], [26, 2], [26, 6], [26, 10], [26, 14], [26, 18], [26, 22], [26, 26], [26, 30], [26, 34], [26, 38], [26, 42], [26, 46], [26, 50], [26, 54], [26, 58], [26, 62], [26, 66], [26, 70], [26, 74], [26, 78], [30, 2], [30, 6], [30, 10], [30, 14], [30, 18], [30, 22], [30, 26], [30, 30], [30, 34], [30, 38], [30, 42], [30, 46], [30, 50], [30, 54],
# #             [30, 58], [30, 62], [30, 66], [30, 70], [30, 74], [30, 78], [34, 2], [34, 6], [34, 10], [34, 14], [34, 18], [34, 22], [34, 26], [34, 30], [34, 34], [34, 38], [34, 42], [34, 46], [34, 50], [34, 54], [34, 58], [34, 62], [34, 66], [34, 70], [34, 74], [34, 78], [38, 2], [38, 6], [38, 10], [38, 14], [38, 18], [38, 22], [38, 26], [38, 30], [38, 34], [38, 38], [38, 42], [38, 46], [38, 50], [38, 54], [38, 58], [38, 62], [38, 66], [38, 70], [38, 74], [38, 78], [42, 2], [42, 6], [42, 10], [42, 14],
# #             [42, 18], [42, 22], [42, 26], [42, 30], [42, 34], [42, 38], [42, 42], [42, 46], [42, 50], [42, 54], [42, 58], [42, 62], [42, 66], [42, 70], [42, 74], [42, 78], [46, 2], [46, 6], [46, 10], [46, 14], [46, 18], [46, 22], [46, 26], [46, 30], [46, 34], [46, 38], [46, 42], [46, 46], [46, 50], [46, 54], [46, 58], [46, 62], [46, 66], [46, 70], [46, 74], [46, 78], [50, 2], [50, 6], [50, 10], [50, 14], [50, 18], [50, 22], [50, 26], [50, 30], [50, 34], [50, 38], [50, 42], [50, 46], [50, 50], [50, 54],
# #             [50, 58], [50, 62], [50, 66], [50, 70], [50, 74], [50, 78], [54, 2], [54, 6], [54, 10], [54, 14], [54, 18], [54, 22], [54, 26], [54, 30], [54, 34], [54, 38], [54, 42], [54, 46], [54, 50], [54, 54], [54, 58], [54, 62], [54, 66], [54, 70], [54, 74], [54, 78], [58, 2], [58, 6], [58, 10], [58, 14], [58, 18], [58, 22], [58, 26], [58, 30], [58, 34], [58, 38], [58, 42], [58, 46], [58, 50], [58, 54], [58, 58], [58, 62], [58, 66], [58, 70], [58, 74], [58, 78], [62, 2], [62, 6], [62, 10], [62, 14],
# #             [62, 18], [62, 22], [62, 26], [62, 30], [62, 34], [62, 38], [62, 42], [62, 46], [62, 50], [62, 54], [62, 58], [62, 62], [62, 66], [62, 70], [62, 74], [62, 78], [66, 2], [66, 6], [66, 10], [66, 14], [66, 18], [66, 22], [66, 26], [66, 30], [66, 34], [66, 38], [66, 42], [66, 46], [66, 50], [66, 54], [66, 58], [66, 62], [66, 66], [66, 70], [66, 74], [66, 78], [70, 2], [70, 6], [70, 10], [70, 14], [70, 18], [70, 22], [70, 26], [70, 30], [70, 34], [70, 38], [70, 42], [70, 46], [70, 50], [70, 54],
# #             [70, 58], [70, 62], [70, 66], [70, 70], [70, 74], [70, 78], [74, 2], [74, 6], [74, 10], [74, 14], [74, 18], [74, 22], [74, 26], [74, 30], [74, 34], [74, 38], [74, 42], [74, 46], [74, 50], [74, 54], [74, 58], [74, 62], [74, 66], [74, 70], [74, 74], [74, 78], [78, 2], [78, 6], [78, 10], [78, 14], [78, 18], [78, 22], [78, 26], [78, 30], [78, 34], [78, 38], [78, 42], [78, 46], [78, 50], [78, 54], [78, 58], [78, 62], [78, 66], [78, 70], [78, 74], [78, 78]]
# #
# # raw = [[24.0, 35.0], [68.5, 54.0], [69.5, 37.5], [39.0, 45.5], [41.0, 30.0], [38.5, 25.0], [31.5, 21.5], [37.0, 15.0]]
# #
# # fin = []
# # for r in raw:
# #     li = []
# #     for d in dot_list:
# #         dis1 = abs(r[0]-d[0])
# #         dis2 = abs(r[1]-d[1])
# #         li.append(dis1+dis2)
# #     # mn = li.index(min(li))
# #     dis, idx = li[0], 0
# #     for i in range(0, len(li)):
# #         if li[i] < dis:
# #             dis = li[i]
# #             idx = i
# #     fin.append(idx)
#
# fin = [[1,3,5,7], [2,4,6,8,10]]
# label_padding = np.zeros([2,11,1], dtype=int)
# for idx, ele in enumerate(fin):
#     for e in ele:
#         label_padding[idx, e] = 1
#
# print(label_padding)
#
#


num = 0

def inc(i):
    global num
    i += 1
    num = i
    return i

print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
num = 0
print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
num = 0
print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
print(inc(num))
