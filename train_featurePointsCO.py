from dataprocess.TeethDataLoader_ver10 import TeethDataLoader
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
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
result_savepath = './log/result/result_CO/'


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet_muti_r')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='all_teeth_ver11', help='model name [default: pointnet2_completion_msg]')
    parser.add_argument('--epoch',  default=401, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.002, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=0, help='decay rate [default: 1e-4]')
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

def test(model, loader):
    mean_correct = []
    mean_loss = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, features, _, seg_gt, _, _ = data

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
        points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
        points = torch.Tensor(points)

        '''heatmap num'''
        heatmap_gt = features[:,:,0]

        '''seg_gt'''
        seg_gt = vert2matrix(seg_gt)

        points = points.transpose(2, 1)
        # points = points.cuda()

        '''heatmap num'''
        # heatmap_gt = heatmap_gt.cuda()

        # seg_gt = torch.Tensor(seg_gt).transpose(2, 1).cuda()
        seg_gt = torch.Tensor(seg_gt).transpose(2, 1)

        classifier = model.eval()
        heatmap = classifier(points, seg_gt)

        '''heatmap num'''
        lossH = F.mse_loss(heatmap.float(), heatmap_gt.float())

        '''heatmap num'''
        loss = lossH

        mean_loss.append(loss)
    
    instance_acc = torch.mean(torch.stack(mean_loss))
    return instance_acc



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
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data_1000/teeth_data_4096_withSeg_RBF/'

    TRAIN_DATASET = TeethDataLoader(root=DATA_PATH, npoint=args.num_point, split='train',
                                                     normal_channel=args.normal)
    TEST_DATASET = TeethDataLoader(root=DATA_PATH, npoint=args.num_point, split='test',
                                                    normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    # 读取模型 14类
    # classifier = MODEL.get_model(14).cuda()
    # criterion = MODEL.get_loss().cuda()
    classifier = MODEL.get_model(14)
    criterion = MODEL.get_loss()

    # try:
    #     checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #     start_epoch = checkpoint['epoch']
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     log_string('Use pretrain model')
    # except:
    #     log_string('No existing model, starting training from scratch...')
    #     start_epoch = 0

    def weights_init(m): # 参数初始化
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load('./log/temp_checkpoint/best_modelCO.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    
    


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7, last_epoch=-1)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 10000.0
    best_class_acc = 0.0
    mean_correct = []
    mean_loss = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # 这是之前dataloader读的数据
            points, features, _, seg_gt, _, _ = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            
            '''heatmap num'''
            heatmap_gt = features[:,:,0]

            '''seg_gt'''
            seg_gt = vert2matrix(seg_gt)

            points = points.transpose(2, 1)

            # points = points.cuda()

            '''heatmap num'''
            # heatmap_gt = heatmap_gt.cuda()

            # seg_gt = torch.Tensor(seg_gt).transpose(2, 1).cuda()
            seg_gt = torch.Tensor(seg_gt).transpose(2, 1)

            optimizer.zero_grad()

            classifier = classifier.train()

            '''heatmap num'''
            heatmap = classifier(points,seg_gt)
            loss = criterion(heatmap.float(), heatmap_gt.float())

            mean_loss.append(loss)
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = torch.mean(torch.stack(mean_loss))
        log_string('Train Loss: %f' % train_instance_acc)


        with torch.no_grad():
            instance_acc= test(classifier.eval(), testDataLoader)

            if (instance_acc <= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            
            log_string('Test Loss: %f'% instance_acc)
            log_string('Best Loss: %f'% best_instance_acc)

            if (instance_acc <= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_modelCO.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                torch.save(state,result_savepath+'best_modelCO.pth')
            
            if(epoch % 10 == 0):
                logger.info('Save model...')
                savepath = './log/temp_checkpoint/latest_modelCO.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': epoch,
                    'instance_acc': instance_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                torch.save(state,result_savepath+'latest_modelCO.pth')

            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
