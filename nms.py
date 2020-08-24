#coding:utf-8  
import numpy as np  
import pickle
from tqdm import tqdm

def py_cpu_nms(dets, thresh=0.5):  
    """Pure Python NMS baseline."""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4]  #bbox打分
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    #打分从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
        #order[0]是当前分数最大的窗口，肯定保留  
        i = order[0]  
        keep.append(i)  
        #计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]  
  
    return keep

if __name__=='__main__':
    pkl_data = pickle.load(open('./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico.pkl','rb'))
    det_data = pickle.load(open('./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico_Ap30.8_cleanFormat.pkl','rb'))

    pkl_img1_data = []
    det_img1_data = []

    for data in pkl_data[1]:
        bbox = np.array([data[2][0], data[2][1], data[2][2], data[2][3], data[5]])
        pkl_img1_data.append(bbox)

    pkl_img1_data = np.array(pkl_img1_data)
    print(pkl_img1_data.shape)

    pkl_keep = py_cpu_nms(pkl_img1_data, thresh=0.5)
    pkl_keep = np.array(pkl_keep)
    print(pkl_keep.shape)