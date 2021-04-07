import random
import numpy as np
import math
from Kmeans import kmeans
 
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

'''
CO : 
CU : all bad 
FA : 0.6
OC : 0.75

'''
if __name__ == '__main__' : 
    point_set = np.loadtxt('model_1.txt')
    new_points = []
    for p in point_set:
        if p[3] > 0.37:
            new_points.append(p)
    # np.savetxt('testdbscan.txt',new_points)
    c = dbscan(new_points, 1, 3)
    idx = []
    for i in range(len(c)):
        if c[i] != -1:
            idx.append(i)
    c = np.array(c)
    c = c[idx]
    new_points = np.array(new_points)
    new_points = new_points[idx,:]
    new_points = new_points[:, 0:3]
    n = len(c)
    k = np.max(c)+1
    rgb = np.zeros(shape = (n , 3))
    for i in range(n):
        seg = c[i]
        rgb[i][0] = 1 - 1 * seg/k
        rgb[i][1] = 0
        rgb[i][2] = 0
    new_points = np.hstack((new_points,rgb))
    print(k)
    np.savetxt('testdbscan.txt',new_points)

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
    
    rgb = np.zeros(shape = (k , 3))
    for r in rgb:
        r[1] = 1
    cent = np.hstack((cent , rgb))
    point_set = point_set[:,0:3]
    N =point_set.shape[0]
    rgb = np.zeros(shape = (N , 3))
    for r in rgb:
        r[0] = 0.5
        r[1] = 0.5
        r[2] = 0.5
    point_set = np.hstack((point_set , rgb))
    point_set = np.vstack((point_set,cent))
    np.savetxt('testkmean.txt',cent)
    np.savetxt('Visualization/pred_data/model_1-PredCO.txt',point_set)
    # point_set = np.loadtxt('data_1000/teeth_data_RBF/model_1.txt')
    # new_points = []
    # for p in point_set:
    #     if p[3] > 0.6:
    #         new_points.append(p)
    # # np.savetxt('testdbscan.txt',new_points)
    # c = dbscan(new_points, 1, 2)
    # new_points = np.array(new_points)
    # new_points = new_points[:, 0:3]
    # n = len(c)
    # rgb = np.zeros(shape = (n , 3))
    # for i in range(n):
    #     seg = c[i]
    #     rgb[i][0] = 255 - 255 * seg/13
    #     rgb[i][1] = 0
    #     rgb[i][2] = 0
    # new_points = np.hstack((new_points,rgb))
    # k = np.max(c)+1
    # print(k)
    # np.savetxt('testdbscan.txt',new_points)
    # cent = np.zeros(shape = (14 , 6))
    # for i in range(n):
    #     cent[c[i]] += new_points[i]
    # cent = cent/14
    # for p in cent:
    #     p[3] = 255
    # new_points = np.vstack((new_points, cent))
    # np.savetxt('testdbscanMean.txt',new_points)

    # kpoints = [] #将点按聚类分成k类
    # for i in range(14): 
    #     p = []
    #     kpoints.append(p)
    # for i in range(n):
    #     kpoints[c[i]].append(new_points[i][0:3])

    # cent = []
    # for k in kpoints: #对每一类的全部点使用kmeans求一个聚类中心
    #     centroids,_ = kmeans(k, 1)
    #     cent.append(centroids)
    # cent = np.array(cent)
    # cent = cent.reshape(14,-1)
    
    # rgb = np.zeros(shape = (14 , 3))
    # for r in rgb:
    #     r[0] = 255
    # cent = np.hstack((cent , rgb))
    # new_points = np.vstack((new_points,cent))
    # np.savetxt('testdbscanMean.txt',new_points)


