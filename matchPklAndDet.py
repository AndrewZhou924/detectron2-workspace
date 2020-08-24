import pickle
import numpy as np
from tqdm import tqdm

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
 
 
if __name__=='__main__':
    # rect1 = (661, 27, 679, 47)
    # (top, left, bottom, right)
    # rect2 = (662, 27, 682, 47)
    # iou = compute_iou(rect1, rect2)
    # print(iou)

    pkl_data = pickle.load(open('./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico.pkl','rb'))
    det_data = pickle.load(open('./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico_Ap30.8_cleanFormat.pkl','rb'))

    cnt_match   = 0
    cnt_unmatch = 0
    cnt_BBox    = 0
    for key in tqdm(pkl_data.keys()):
        for i in range(len(det_data[key])):
            cnt_BBox += 1
            det_BBox = det_data[key][i][2]
            match_flag = False

            for j in range(len(pkl_data[key])):
                pkl_BBox = pkl_data[key][j][2]
                iou      = compute_iou(det_BBox, pkl_BBox)
                # print(iou, det_BBox, pkl_BBox)

                if iou > 0.9:
                    match_flag = True
                    break


            if match_flag == False:
                cnt_unmatch += 1
            else:
                cnt_match += 1

    print("==> cnt_BBox = {}".format(cnt_BBox))
    print("==> match = {}, unmatch = {}".format(cnt_match, cnt_unmatch))

    '''
    0.95
    cnt_BBox = 107341
    match = 103318, unmatch = 4023 96.2%

    0.9
    match = 104438, unmatch = 2903 97.3%

    0.8
    match = 105320, unmatch = 2021 98.1%
    '''