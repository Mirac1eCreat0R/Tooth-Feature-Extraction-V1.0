import numpy as np
def generateHeat(feature, in_file, out_file):
    feature = feature + 3
    point_set = np.loadtxt(in_file)
    point_set = point_set[np.argsort(point_set[:,feature])]

    featmax = point_set.max(axis=0)[feature]
    featmin = point_set.min(axis=0)[feature]

    print("max" + str(featmax))
    print("min" + str(featmin))
    
    for point in point_set:
        point[feature] = (point[feature] - featmin) / (featmax - featmin)
    point_set = point_set[:,[0,1,2,feature]]
    n = point_set.shape[0]
    rgb = np.empty(shape = (n , 3))
    point_set = np.hstack((point_set , rgb))
    # for point in point_set:
    #     point[4] = 255 - 255 * point[3]
    #     point[5] = 0
    #     point[6] = 255 * point[3]
    for point in point_set:
         point[4] = point[3]*1
         point[5] = 0
         point[6] = 1 - point[3]*1
    point_set = np.delete(point_set, 3, axis=1)
    np.savetxt(out_file, point_set)

if __name__ == '__main__':
    feature = 0
    # in_file = 'Visualization/pred_data/model_1-OC.txt'
    in_file = 'data_1000/teeth_data_4096_withSeg_RBF_Single_new/model_76.txt'
    out_file = '1.txt'
    generateHeat(feature, in_file, out_file)