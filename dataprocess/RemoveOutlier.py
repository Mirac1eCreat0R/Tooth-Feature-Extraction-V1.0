import numpy as np
import random
import math
import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

point_set = np.loadtxt('data_1000/teeth_847_withSeg/model_1.txt')
point_set = point_set[:,[0,1,2,7]]
point_set_new = np.empty(shape=(0,4))
segdict = {}
for p in point_set:
    seg = int(p[3])
    if seg in segdict.keys():
        segdict[seg].append(p)
    else:
        segdict[seg] = []
        segdict[seg].append(p)

for key in segdict.keys():
    v = segdict[key]
    new_points = np.array(v)

    n = new_points.shape[0]

    c = DBSCAN(eps=0.8,min_samples=3).fit_predict(new_points[:,0:3])
    c = np.array(c)

    idx = []

    a = np.bincount(c)
    m = np.argmax(a)

    for i in range(len(c)):
        if c[i] == m:
            idx.append(i)
    new_points = new_points[idx,:]
    point_set_new = np.vstack((point_set_new , new_points))

np.savetxt('1.txt' , point_set_new)
