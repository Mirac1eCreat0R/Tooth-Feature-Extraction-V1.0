import numpy as np
import random
import math
import os
import pandas as pd
from sklearn.cluster import SpectralClustering
from Kmeans import kmeans


def calfeatCO(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CO.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.20:
            seg = int(p[4])
            if seg in segdict.keys():
                segdict[seg].append(p)
            else:
                segdict[seg] = []
                segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        new_points = np.array(v)
        new_points = new_points[:, 0:3]

        n = new_points.shape[0]
        rgb = np.zeros(shape = (n , 3))
        new_points = np.hstack((new_points,rgb))

        c = SpectralClustering(n_clusters=2 , gamma=2).fit_predict(new_points[:,0:3])
        c = np.array(c)
        k = np.max(c)+1

        kpoints = [] #将点按聚类分成k类
        for i in range(k): 
            p = []
            kpoints.append(p)
        for i in range(n):
            kpoints[c[i]].append(new_points[i][0:3])

        cent = []
        for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
            centroids,_ = kmeans(kp, 1)
            cent.append(centroids)
        cent = np.array(cent)
        cent = cent.reshape(k,-1)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[1] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.5
        r[1] = 0.5
        r[2] = 0.5
    point_set = np.hstack((point_set , rgb))
    point_set = np.vstack((point_set, cent_points))
    np.savetxt('testkmeanCO.txt',cent_points)
    np.savetxt(name + '-PredCO.txt',point_set)

# def calfeatCU(name):
#     '''
#     带入segment信息进行分割
#     '''
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CU.txt')
#     segdict = {}
#     for p in point_set:
#         if p[3] > 0.25:
#             seg = int(p[4])
#             if seg in segdict.keys():
#                 segdict[seg].append(p)
#             else:
#                 segdict[seg] = []
#                 segdict[seg].append(p)

#     cent_points = np.empty(shape = (0 , 6))
#     for key in segdict.keys():
#         v = segdict[key]
#         new_points = np.array(v)
#         new_points = new_points[:, 0:3]

#         n = new_points.shape[0]
#         rgb = np.zeros(shape = (n , 3))
#         new_points = np.hstack((new_points,rgb))

#         if key in [11,12,21,22]:
#             cent,_ = kmeans(new_points[:, 0:3], 3)
#         if key in [13,23]:
#             cent,_ = kmeans(new_points[:, 0:3], 1)
#         if key in [14,15,24,25]:
#             cent,_ = kmeans(new_points[:, 0:3], 2)
#         if key in [16,17,26,27]:
#             cent,_ = kmeans(new_points[:, 0:3], 4)
#         cent = np.array(cent)
    
#         rgb = np.zeros(shape = (cent.shape[0] , 3))
#         for r in rgb:
#             r[1] = 1
#         cent = np.hstack((cent , rgb))
#         cent_points = np.vstack((cent_points , cent))
        
#     point_set = point_set[:,0:3]
#     N =point_set.shape[0]
#     rgb = np.zeros(shape = (N , 3))
#     for r in rgb:
#         r[0] = 0.5
#         r[1] = 0.5
#         r[2] = 0.5
#     point_set = np.hstack((point_set , rgb))
#     point_set = np.vstack((point_set, cent_points))
#     np.savetxt('testkmeanCU.txt',cent_points)
#     np.savetxt(name + '-PredCU.txt',point_set)

def gt(name , feature):

    feat = ['CO.txt' , 'CU.txt' , 'FA.txt' , 'OC.txt']
    featurenameW = ['-PredCOW' , '-PredCUW' , '-PredFAW' , '-PredOCW']

    featfilepath = 'data_1000/feature_1000'
    featlist = np.empty(shape = (0,3))

    for featfile in os.listdir(featfilepath):
        idx = featfile.split("_")[1]
        nameidx = name.split("_")[1]
        if idx == nameidx and featfile.endswith(feat[feature]):
            featpoint = np.loadtxt(featfilepath + '/' + featfile)
            featlist = np.vstack((featlist , featpoint))
    n = featlist.shape[0]
    rgb = np.zeros(shape = (n , 3))
    for r in rgb:
        r[0] = 1
        r[2] = 1
    featlist = np.hstack((featlist , rgb))
    np.savetxt('gtfile.txt' , featlist)

if __name__ == '__main__':
    calfeatCO('model_1')
    gt('model_1' , 0)