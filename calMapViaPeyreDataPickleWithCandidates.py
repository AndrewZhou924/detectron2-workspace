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
            default='./data/db_test.pkl', type=str)
    
    parser.add_argument('--candidates', dest='candidates', 
            help='candidates pairs',
            default='./data/candidates_test.pkl', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arg parsing
    args = parse_args()
    print("==> result file: ", args.result)
    category_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, \
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    # loading data
    hicoGroundTruth  = "./data/HICO-DET-Detector/hico_annotations_test2015.json"
    cocoGt           = COCO(hicoGroundTruth) #取得标注集中coco json对象
    ImgIds           = cocoGt.getImgIds()

    # preparing Det data
    pklData     = pickle.load(open(args.result, 'rb'))
    candidates  = pickle.load(open(args.candidates, 'rb'))
    candidates_dict = {}
    for pair in candidates:
        [key,pairId] = pair
        if key not in candidates_dict.keys():
            candidates_dict[key] = [pairId]
        else:
            candidates_dict[key].append(pairId)
    
    print(len(list(candidates_dict.keys())))
    # exit()
    
    annotations = []
    cat_list    = []
    unMatchId   = []
    for key,imgData in tqdm(pklData.items()):
        
        img_id = int(imgData['filename'].strip('.jpg').split('_')[-1])
        try:
            candidates_pairs = candidates_dict[key]
        except:
            print("can't match key={} in pklData".format(key))
            unMatchId.append(key)
            continue
        
        boxIndex = []
        for pairId in candidates_pairs:
            box1,box2 = imgData['pair_ids'][pairId]
            boxIndex.append(box1)
            boxIndex.append(box2)
        boxIndex = set(boxIndex)
        
        # for i in range(len(imgData['obj_classes'])):
        for i in boxIndex:
            
            if imgData['is_gt'][i]: # skip gt bbox
                continue

            [x1,y1,x2,y2] = imgData['boxes'][i]
            x1            = min(x1,x2)
            x2            = max(x1,x2)
            y1            = min(y1,y2)
            y2            = max(y1,y2)
            width         = abs(x2 - x1)
            height        = abs(y2 - y1)
            category      = imgData['obj_classes'][i]
            category_id   = category_set[category-1]
            score         = imgData['obj_scores'][i]

            anno = {
                "image_id": img_id, 
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
    
    print("len(unMatchId):", len(unMatchId))
    # print("len={}\n".format(len(set(cat_list))), set(cat_list))
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

    # process annotations_GT
    # change box format from XYXY to XYWH (and adding area)
    # tmp_annos = []
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
        # tmp_annos.append(anno)

    '''
    # JUST FOR TEST
    cocoDt = COCO()
    cocoDt.dataset['images']      = [img for img in cocoGt.dataset['images']]
    cocoDt.dataset['categories']  = copy.deepcopy(cocoGt.dataset['categories'])
    for idx,anno in enumerate(tmp_annos):
        tmp_annos[idx]['score'] = 1.0
    cocoDt.dataset['annotations'] = tmp_annos
    cocoDt.createIndex()
    '''

    # start evaluation
    cocoEval     = COCOeval(cocoGt, cocoDt, 'bbox') 

    # select N imgs to evaluate
    # imgIds_sort  = sorted(ImgIds)
    # cocoEval.params.imgIds = imgIds_sort[:500]

    cocoEval.evaluate()    #评价
    cocoEval.accumulate()  #积累
    cocoEval.summarize()   #总结

    exit()
