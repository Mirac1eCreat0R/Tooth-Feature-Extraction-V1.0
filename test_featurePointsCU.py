from data_utils.TeethDataLoader_ver10 import TeethDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet_muti_r')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size in testing [default: 24]')
    parser.add_argument('--model', default='all_teeth_ver11', help='model name [default: pointnet2_completion_msg]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default=None, help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def vert2matrix(inputVert):
    '''
    transform labels from vert to matrix
    '''
    inputVert = inputVert.numpy()
    B,N = inputVert.shape
    outputMat = np.zeros(shape = (B, N, 14))
    # print(outputMat)
    for b in range(B):
        for i in range(N):
            seg = inputVert[b,i]
            m = 19 - int(seg)
            if m > 0:
                p = 8 - m
                outputMat[b,i,p] = 1
            else:
                m = -1 * m
                p = m + 5
                outputMat[b,i,p] = 1
    return outputMat

def vert2vert(inputVert):
    '''
    transform labels from 11 12 13 14 15 16 17 21 22 23 24 25 26 27 to 0-13
    '''
    inputVert = inputVert.numpy()
    B,N = inputVert.shape
    outputMat = np.zeros(shape = (B, N))
    # print(outputMat)
    for b in range(B):
        for i in range(N):
            seg = inputVert[b,i]
            m = 19 - int(seg)
            if m > 0:
                p = 8 - m
                outputMat[b,i] = p
            else:
                m = -1 * m
                p = m + 5
                outputMat[b,i] = p
    return outputMat

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('feature_extract_TEST')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    savepath = './data_1000/result/result_CU/'

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data_1000/teeth_data_4096_withSeg_RBF/'

    TEST_DATASET = TeethDataLoader(root = DATA_PATH, npoint=args.num_point, split='all', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))

    '''MODEL LOADING'''
    # model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(14).cuda()
    checkpoint = torch.load('./log/result/result_CU/latest_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        correct = 0
        mean_correct = 0
        for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            points, features, model_name, point_label, m, centroid = data

            points = points.data.numpy()
            # points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            points = points.cuda()
            seg_gt = vert2matrix(point_label)

            seg_gt = torch.Tensor(seg_gt).transpose(2, 1).cuda()
            heatmap = classifier(points, seg_gt)

            points = points.transpose(2, 1)
            heatmap = heatmap.transpose(2, 1)

            points = points.cpu()
            heatmap = heatmap.cpu()

            m = m.cpu()
            centroid = centroid.cpu()

            for i, model in enumerate(model_name):
                xyz = points[i,:,0:3].numpy()*m[i].numpy()+centroid[i].numpy()
                heatmap_ = heatmap[i,:,:].numpy()


                point_label_i = point_label[i,:].numpy()
                point_label_i = point_label_i.reshape(len(point_label_i),1)

                np.savetxt(savepath+model+'.txt',np.concatenate((xyz,heatmap_,point_label_i),axis=1))

if __name__ == '__main__':
    args = parse_args()
    main(args)

