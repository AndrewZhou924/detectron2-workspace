import pickle
import numpy as np
import copy
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# loading data
hicoGroundTruth = "./data/HICO-DET-Detector/hico_annotations_test2015.json"
VCLDetOutputJson = "./codeInDetectron2OriRepo/output/inference/coco_instances_results.json"
cocoGt = COCO(hicoGroundTruth)  # 取得标注集中coco json对象
category_set = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, \
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, \
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
        76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)

for key, anno in cocoGt.anns.items():
    [x1, y1, x2, y2] = anno['bbox']
    x1 = min(x1, x2)
    x2 = max(x1, x2)
    y1 = min(y1, y2)
    y2 = max(y1, y2)

    width = abs(x2 - x1)
    height = abs(y2 - y1)
    # anno['bbox']  = [x1, y1, width, height] Don't do that
    anno['area'] = width * height
    cocoGt.anns[key] = anno

# start evaluation
cocoDt = cocoGt.loadRes(VCLDetOutputJson)
cocoDt.createIndex()

'''
pkl_data
0: image_id
1: 'Human' or 'Object'
2: BBox
3: nan
4: class id
5: score
'''
data = dict()
cat_list = []
for key,ann in tqdm(cocoDt.anns.items()):
    pkl_data = {}
    image_id    = ann['image_id']
    category_id = category_set.index(ann['category_id']) + 1
    # print(ann['category_id'], category_id)
    cat_list.append(category_id)

    pkl_data[0] = image_id # 0

    if category_id == 1:
        pkl_data[1] = 'Human'
    else:
        pkl_data[1] = 'Object'

    pkl_data[2] = ann['bbox']
    pkl_data[3] = None
    pkl_data[4] = category_id
    pkl_data[5] = ann['score']

    if image_id in data.keys():
        data[image_id].append(pkl_data)
    else:
        data[image_id] = [pkl_data]

cat_list = set(cat_list)
print(len(cat_list))
print(cat_list)

f = open('./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico_Ap30.8.pkl', 'wb')
pickle.dump(data, f)