import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

'''
需要读取附带牙齿分类信息的数据4096_withSeg
'''

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class TeethDataLoader(Dataset):
    def __init__(self, root,  npoint=4096, split='train', uniform=False, normal_channel=False, cache_size=1500):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform

        # self.cat = [line.rstrip() for line in open(self.catfile)]
        # self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'teeth_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'teeth_test.txt'))]
        shape_ids['all'] = [line.rstrip() for line in open(os.path.join(self.root, 'teeth_all.txt'))]

        assert (split == 'train' or split == 'test' or split == 'all')
        names = [x for x in shape_ids[split]]
        self.model_names = names
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [self.root + names[i] + '.txt' for i in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, heatmap_gt, model_name, point_label, m, centroid = self.cache[index]
        else:
            f = self.datapath[index]
            # 取出数据
            data = np.loadtxt(f)
            # 取出坐标信息
            point_set = data[:, 0:3]
             #取出groundtruth的heatmap信息,分别是CO, CU, FA, OC
            heatmap_gt = data[:, 3:7]
            model_name = self.model_names[index]
            point_label = data[:,7]
            point_set,m,centroid = pc_normalize(point_set)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, heatmap_gt, model_name, point_label, m, centroid) 
            
        # 归一化 方法返回归一化参数 用来算回原坐标
        
        return point_set, heatmap_gt, model_name, point_label, m, centroid

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = TeethDataLoader('data_1000/teeth_data_4096_withSeg/',split='train', uniform=False, normal_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,target,model_name, point_label, m, centroid in DataLoader:
        print(vert2matrix(point_label))