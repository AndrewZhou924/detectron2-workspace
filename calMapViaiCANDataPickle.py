import pickle
import numpy as np
import copy
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval

'''
Reference
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
'''

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation for detector on HICO-Det testset')
    parser.add_argument('--result', dest='result', 
            help='detection result on hico test',
            default="/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl", type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    print("==> result file: ", args.result)
    category_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, \
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)

    # loading data
    hicoGroundTruth  = "./data/HICO-DET-Detector/hico_annotations_test2015.json"
    cocoGt           = COCO(hicoGroundTruth) #取得标注集中coco json对象
    ImgIds           = cocoGt.getImgIds()

    # '''
    # preparing Det data
    pklData     = pickle.load(open(args.result, 'rb'), encoding="bytes")
    annotations = []
    cat_list    = []
    
    for key, imgData in tqdm(pklData.items()):

        for idx, info in enumerate(imgData):
            [x1,y1,x2,y2] = info[2]
            x1            = min(x1,x2)
            x2            = max(x1,x2)
            y1            = min(y1,y2)
            y2            = max(y1,y2)
            width         = abs(x2 - x1)
            height        = abs(y2 - y1)
            category      = info[4]
            category_id   = category_set[category-1]
            score         = info[5]

            anno = {
                "image_id": key, 
                "category_id": category_id,
                'area': width * height,
                'bbox': [x1, y1, width, height],
                "iscrowd": 0, 
                "id": 0, 
                "score": score,
                }
            annotations.append(anno)
            
            # print(anno)
            cat_list.append(category_id)
    

    # print(set(cat_list))
    # exit()

    for idx, ann in enumerate(annotations):
        ann['id'] = idx + 1

    annsImgIds = [ann['image_id'] for ann in annotations]
    assert set(annsImgIds) == (set(annsImgIds) & set(cocoGt.getImgIds()))

    cocoDt = COCO()
    cocoDt.dataset['images']      = [img for img in cocoGt.dataset['images']]
    cocoDt.dataset['categories']  = copy.deepcopy(cocoGt.dataset['categories'])
    cocoDt.dataset['annotations'] = annotations
    cocoDt.createIndex()
    # '''

    # process annotations_GT
    # change box format from XYXY to XYWH (and adding area)
    for key, anno in cocoGt.anns.items():
        [x1,y1,x2,y2] = anno['bbox']
        x1            = min(x1,x2)
        x2            = max(x1,x2)
        y1            = min(y1,y2)
        y2            = max(y1,y2)
        width         = abs(x2 - x1)
        height        = abs(y2 - y1)

        # Don't do that
        # anno['bbox']   = [x1, y1, width, height] 
        anno['area']     = width * height
        cocoGt.anns[key] = anno

    # start evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox') 

    # select N imgs to evaluate
    imgIds_sort  = sorted(ImgIds)
    cocoEval.params.imgIds = imgIds_sort[:500]

    cocoEval.evaluate()    #评价
    cocoEval.accumulate()  #积累
    cocoEval.summarize()   #总结

    exit()
