import numpy as np
import random
import math
import os
import pandas as pd

# 计算两个点之间的欧式距离，参数为两个元组
def dist(t1, t2):
    dis = math.sqrt(np.power((t1[0]-t2[0]),2) + np.power((t1[1]-t2[1]),2) + np.power((t1[2]-t2[2]),2))
    return dis
 
# DBSCAN算法，参数为数据集，Eps为指定半径参数，MinPts为制定邻域密度阈值
def dbscan(Data, Eps, MinPts):
    num = len(Data)  # 点的个数
    # print("点的个数："+str(num))
    unvisited = [i for i in range(num)]  # 没有访问到的点的列表
    # print(unvisited)
    visited = []  # 已经访问的点的列表
    C = [-1 for i in range(num)]
    # C为输出结果，默认是一个长度为num的值全为-1的列表
    # 用k来标记不同的簇，k = -1表示噪声点
    k = -1
    # 如果还有没访问的点
    while len(unvisited) > 0:
        # 随机选择一个unvisited对象
        p = random.choice(unvisited)
        unvisited.remove(p)
        visited.append(p)
        # N为p的epsilon邻域中的对象的集合
        N = []
        for i in range(num):
            if (dist(Data[i], Data[p]) <= Eps):# and (i!=p):
                N.append(i)
        # 如果p的epsilon邻域中的对象数大于指定阈值，说明p是一个核心对象
        if len(N) >= MinPts:
            k = k+1
            # print(k)
            C[p] = k
            # 对于p的epsilon邻域中的每个对象pi
            for pi in N:
                if pi in unvisited:
                    unvisited.remove(pi)
                    visited.append(pi)
                    # 找到pi的邻域中的核心对象，将这些对象放入N中
                    # M是位于pi的邻域中的点的列表
                    M = []
                    for j in range(num):
                        if (dist(Data[j], Data[pi])<=Eps): #and (j!=pi):
                            M.append(j)
                    if len(M)>=MinPts:
                        for t in M:
                            if t not in N:
                                N.append(t)
                # 若pi不属于任何簇，C[pi] == -1说明C中第pi个值没有改动
                if C[pi] == -1:
                    C[pi] = k
        # 如果p的epsilon邻域中的对象数小于指定阈值，说明p是一个噪声点
        else:
            C[p] = -1
 
    return C

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist=[]
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids  #相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2     #平方
        squaredDist = np.sum(squaredDiff, axis=1)   #和  (axis=1表示行)
        distance = squaredDist ** 0.5  #开根号
        clalist.append(distance) 
    clalist = np.array(clalist)  #返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist

# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)    #axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean() #DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values
 
    # 计算变化量
    changed = newCentroids - centroids
 
    return changed, newCentroids

# 使用k-means分类
def kmeans(dataSet, k):
    # 随机取质心
    centroids = random.sample(list(dataSet), k)
    
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
 
    centroids = sorted(newCentroids.tolist())   #tolist()将矩阵转换成列表 sorted()排序
 
    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k) #调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)  
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):   #enumerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        
    return centroids, cluster

# def calfeatCO(name):
#     '''
#     带入segment信息进行分割
#     '''
#     point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CO.txt')
#     segdict = {}
#     for p in point_set:
#         if p[3] > 0.20:
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

#         cent,_ = kmeans(new_points[:, 0:3], 2)
#         cent = np.array(cent)
    
#         rgb = np.zeros(shape = (cent.shape[0] , 3))
#         for r in rgb:
#             r[0] = 1
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

#     ncent = cent_points.shape[0]
#     cent_eh = np.zeros(shape = (ncent * 24 , 6))
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + i*ncent][0] = cent_points[j][0]
#             cent_eh[j + i*ncent][1] = cent_points[j][1]
#             cent_eh[j + i*ncent][2] = cent_points[j][2]
#             cent_eh[j + i*ncent][i] = cent_points[j][i] + 0.1
#             cent_eh[j + i*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (3 + i)*ncent][0] = cent_points[j][0] 
#             cent_eh[j + (3 + i)*ncent][1] = cent_points[j][1] 
#             cent_eh[j + (3 + i)*ncent][2] = cent_points[j][2] 
#             cent_eh[j + (3 + i)*ncent][i] = cent_points[j][i] - 0.1
#             cent_eh[j + (3 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (6 + i)*ncent][0] = cent_points[j][0] + 0.1
#             cent_eh[j + (6 + i)*ncent][1] = cent_points[j][1] + 0.1
#             cent_eh[j + (6 + i)*ncent][2] = cent_points[j][2] + 0.1
#             cent_eh[j + (6 + i)*ncent][i] = cent_points[j][i] - 0.1
#             cent_eh[j + (6 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (9 + i)*ncent][0] = cent_points[j][0] - 0.1
#             cent_eh[j + (9 + i)*ncent][1] = cent_points[j][1] - 0.1
#             cent_eh[j + (9 + i)*ncent][2] = cent_points[j][2] - 0.1
#             cent_eh[j + (9 + i)*ncent][i] = cent_points[j][i] + 0.1
#             cent_eh[j + (9 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (12 + i)*ncent][0] = cent_points[j][0]
#             cent_eh[j + (12 + i)*ncent][1] = cent_points[j][1]
#             cent_eh[j + (12 + i)*ncent][2] = cent_points[j][2]
#             cent_eh[j + (12 + i)*ncent][i] = cent_points[j][i] + 0.05
#             cent_eh[j + (12 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (15 + i)*ncent][0] = cent_points[j][0] 
#             cent_eh[j + (15 + i)*ncent][1] = cent_points[j][1] 
#             cent_eh[j + (15 + i)*ncent][2] = cent_points[j][2] 
#             cent_eh[j + (15 + i)*ncent][i] = cent_points[j][i] - 0.05
#             cent_eh[j + (15 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (18 + i)*ncent][0] = cent_points[j][0] + 0.05
#             cent_eh[j + (18 + i)*ncent][1] = cent_points[j][1] + 0.05
#             cent_eh[j + (18 + i)*ncent][2] = cent_points[j][2] + 0.05
#             cent_eh[j + (18 + i)*ncent][i] = cent_points[j][i] - 0.05
#             cent_eh[j + (18 + i)*ncent][3] = 1
#     for i in range(3):
#         for j in range(len(cent_points)):
#             cent_eh[j + (21 + i)*ncent][0] = cent_points[j][0] - 0.05
#             cent_eh[j + (21 + i)*ncent][1] = cent_points[j][1] - 0.05
#             cent_eh[j + (21 + i)*ncent][2] = cent_points[j][2] - 0.05
#             cent_eh[j + (21 + i)*ncent][i] = cent_points[j][i] + 0.05
#             cent_eh[j + (21 + i)*ncent][3] = 1
#     point_set = np.vstack((point_set, cent_points))
#     point_set = np.vstack((point_set, cent_eh))

#     np.savetxt('testkmeanCO.txt',cent_points)
#     np.savetxt(name + '-PredCO.txt',point_set)
def closestPoint(point , point_set):
    n = point_set.shape[0]
    point = point.reshape(-1,3)
    cent = np.empty(shape=(0,3))
    for p in point:
        p = p.reshape(-1,3)
        p = np.repeat(p, n , axis=0)   
        dis = (point_set - p) * (point_set - p)
        dis = np.sum(dis , axis=1)
        dis = np.sqrt(dis)
        i = np.argmin(dis)
        cent_p = point_set[i]
        cent = np.vstack((cent,cent_p))
    return cent

def calfeatCO(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CO.txt')
    segdict = {}
    for p in point_set:
        seg = int(p[4])
        if seg in segdict.keys():
            segdict[seg].append(p)
        else:
            segdict[seg] = []
            segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        v = np.array(v)
        new_points = v[v[:,3].argsort()]
        new_points = new_points[-50:, 0:3]

        cent,_ = kmeans(new_points, 2)
        cent = np.array(cent)
        cent = closestPoint(cent , new_points)
    
        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[0] = 1
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

    point_set = np.vstack((point_set, cent_points))
    # point_set = np.vstack((point_set, cent_eh))

    np.savetxt('testkmeanCO.txt',cent_points)
    np.savetxt(name + '-PredCO.txt',point_set)

def calfeatCU(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-CU.txt')
    segdict = {}
    for p in point_set:
        seg = int(p[4])
        if seg in segdict.keys():
            segdict[seg].append(p)
        else:
            segdict[seg] = []
            segdict[seg].append(p)

    cent_points = np.empty(shape = (0 , 6))
    for key in segdict.keys():
        v = segdict[key]
        v = np.array(v)
        new_points = v[v[:,3].argsort()]
        new_points = new_points[-50:, 0:3]

        if key in [11,12,21,22]:
            cent,_ = kmeans(new_points, 3)
        if key in [13,23]:
            cent,_ = kmeans(new_points, 1)
        if key in [14,15,24,25]:
            cent,_ = kmeans(new_points, 2)
        if key in [16,17,26,27]:
            cent,_ = kmeans(new_points, 4)
        cent = np.array(cent)
        cent = closestPoint(cent , new_points)

        rgb = np.zeros(shape = (cent.shape[0] , 3))
        for r in rgb:
            r[2] = 1
        cent = np.hstack((cent , rgb))
        cent_points = np.vstack((cent_points , cent))
        
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.8
        r[1] = 0.8
        r[2] = 0.8
    point_set = np.hstack((point_set , rgb))
    point_set = np.vstack((point_set, cent_points))
    np.savetxt('testkmeanCU.txt',cent_points)
    np.savetxt(name + '-PredCU.txt',point_set)

def calfeatFA(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-FA.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.155:
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

        cent,_ = kmeans(new_points[:, 0:3], 1)
        cent = np.array(cent)
    
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
    np.savetxt('testkmeanFA.txt',cent_points)
    np.savetxt(name + '-PredFA.txt',point_set)

def calfeatOC(name):
    '''
    带入segment信息进行分割
    '''
    point_set = np.loadtxt('Visualization/pred_data/'+ name + '-OC.txt')
    segdict = {}
    for p in point_set:
        if p[3] > 0.24:
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

        cent,_ = kmeans(new_points[:, 0:3], 2)
        cent = np.array(cent)
    
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
    np.savetxt('testkmeanOC.txt',cent_points)
    np.savetxt(name + '-PredOC.txt',point_set)



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

# def calfeatCO(name):
#     '''
#     带入segment信息进行分割
#     '''
#     point_set = np.loadtxt('data_1000/result/resultv1.1_CO_withSeg/'+ name + '.txt')
#     segdict = {}
#     for p in point_set:
#         if p[3] > 0.23:
#             seg = int(p[4])
#             if seg in segdict.keys():
#                 segdict[seg].append(p)
#             else:
#                 segdict[seg] = []
#                 segdict[seg].append(p)
#     heat_points = np.empty(shape = (0 , 6))
#     cent_points = np.empty(shape = (0 , 6))
#     for k in segdict.keys():
#         v = segdict[k]
#         c = dbscan(v, 1, 3)
#         idx = []
#         for i in range(len(c)):
#             if c[i] != -1:
#                 idx.append(i)
#         c = np.array(c)
#         c = c[idx]
#         new_points = np.array(v)
#         new_points = new_points[idx,:]
#         new_points = new_points[:, 0:3]
#         n = len(c)
#         k = np.max(c)+1
#         rgb = np.zeros(shape = (n , 3))
#         for i in range(n):
#             seg = c[i]
#             rgb[i][0] = 1 - 1 * seg/k
#             rgb[i][1] = 0
#             rgb[i][2] = 0
#         new_points = np.hstack((new_points,rgb))
#         heat_points = np.vstack((heat_points,new_points))

#         kpoints = [] #将点按聚类分成k类
#         for i in range(k): 
#             p = []
#             kpoints.append(p)
#         for i in range(n):
#             kpoints[c[i]].append(new_points[i][0:3])

#         cent = []
#         for kp in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
#             centroids,_ = kmeans(kp, 1)
#             cent.append(centroids)
#         cent = np.array(cent)
#         cent = cent.reshape(k,-1)
    
#         rgb = np.zeros(shape = (k , 3))
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
#     np.savetxt('testdbscan.txt',heat_points)
#     np.savetxt('testkmeanCO.txt',cent_points)
#     np.savetxt(name + '-PredCO.txt',point_set)

if __name__ == '__main__':
    # calfeatCU('model_1')
    gt('model_76' , 1)