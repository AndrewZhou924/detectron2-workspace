import pickle
import numpy as np
import copy
import json
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval

'''
Reference
https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
'''

# loading data
detSourceFile    = "./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico.pkl"
hicoGroundTruth  = "./data/HICO-DET-Detector/hico_annotations_test2015.json"
cocoGt           = COCO(hicoGroundTruth) #取得标注集中coco json对象
detResult        = pickle.load(open(detSourceFile, 'rb')) # 9658
HUMAN_THRESHOLD  = 0.99
OBJECT_THRESHOLD = 0.99

# process annotations_GT
for key, anno in cocoGt.anns.items():
    [x1,y1,x2,y2] = anno['bbox']

    width  = abs(x2 - x1)
    height = abs(y2 - y1)
    x1     = min(x1,x2)
    x2     = max(x1,x2)
    y1     = min(y1,y2)
    y2     = max(y1,y2)

    anno['bbox']  = [x1, y1, width, height]
    anno['area']  = width * height
    cocoGt.anns[key] = anno

# print("GT:", cocoGt.anns[1])
# print("GT:", cocoGt.anns[2])
# exit()

annotations = []
for imgId in detResult.keys():
    
    imgDetRes = detResult[imgId]
    for idx, info in enumerate(imgDetRes):
        [x1,y1,x2,y2] = info[2]
        
        width  = abs(x2 - x1)
        height = abs(y2 - y1)
        x1     = min(x1,x2)
        x2     = max(x1,x2)
        y1     = min(y1,y2)
        y2     = max(y1,y2)

        # print(info[5], [x1, y1, width, height])
        
        # if info[4] == 1 and info[5] < HUMAN_THRESHOLD:
        #     continue
        # elif info[5] < OBJECT_THRESHOLD:
        #     continue

        tmp_anno = {
            "image_id": imgId, 
            "category_id": info[4],
            'area': width * height,
            'bbox': [x1, y1, width, height],
            "iscrowd": 0, 
            "id": 0, 
            "score": info[5],
        }

        annotations.append(tmp_anno)

for idx, ann in enumerate(annotations):
    ann['id'] = idx + 1

# cocoDt = cocoGt.loadRes(detSourceJson) fail
# build res
res = COCO()
# res = copy.deepcopy(cocoGt)
res.dataset['images']      = [img for img in cocoGt.dataset['images']]
res.dataset['categories']  = copy.deepcopy(cocoGt.dataset['categories'])
res.dataset['annotations'] = annotations
res.createIndex()

# print("\nDet:", res.anns[1])
# print("\nDet:", res.anns[2])
# exit()

annsImgIds = [ann['image_id'] for ann in annotations]
assert set(annsImgIds) == (set(annsImgIds) & set(cocoGt.getImgIds()))




cocoEval = COCOeval(cocoGt, res, 'bbox') 

# imgIds  = list(detResult.keys())
# imgIds  = sorted(imgIds)
cocoEval.params.imgIds = annsImgIds #参数设置
cocoEval.evaluate()    #评价
cocoEval.accumulate()  #积累
cocoEval.summarize()   #总结
exit()
