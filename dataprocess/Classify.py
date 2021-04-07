import numpy as np
import os

root = 'data_1000/teeth_data_withSeg/'
for line in os.listdir(root):
    point_set = np.loadtxt(root + line)
    n,_ = point_set.shape
    class_set = np.zeros(shape = (n , 14))
    # print(class_set)
    for i in range(n):
        seg = point_set[i][7]
        m = 19 - int(seg)
        if m > 0:
            p = 8 - m
            class_set[i][p] = 1
        else:
            m = -1 * m
            p = m + 5
            class_set[i][p] = 1
    point_set = np.hstack((point_set, class_set))
    point_set = np.delete(point_set, 7, axis=1)