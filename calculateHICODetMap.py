import pickle
import numpy as np
import copy
import json
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval

# loading data
detSourceFile    = "./data/HICO-DET-Detector/Test_HICO_res101_3x_FPN_hico.pkl"
# detSourceJson  = "./data/HICO-DET-Detector/hico_annotations_det_result.json"
hicoGroundTruth  = "./data/HICO-DET-Detector/hico_annotations_test2015.json"
cocoGt           = COCO(hicoGroundTruth) #取得标注集中coco json对象
detResult        = pickle.load(open(detSourceFile, 'rb'))
# print("==> len(detResult): ", len(detResult)) # 9658

counter = 1
# annotations = dict()
annotations = []
for imgId in detResult.keys():
    
    imgDetRes = detResult[imgId]
    for idx, info in enumerate(imgDetRes):
        bbox     = info[2]
        [x1,y1,x2,y2] = bbox

        width  = max(0, abs(x2 - x1))
        height = max(0, abs(y2 - y1))
        x1 = min(x1,x2)
        y1 = min(y1,y2)

        tmp_anno = {
            "image_id": imgId, 
            "category_id": info[4],
            'area': width * height,
            'bbox': [x1, y1, width, height],
            "iscrowd": 0, 
            "id": (idx+1), 
            "score": info[5]
        }

        # tmp_anno = {"image_id": imgId, "category_id": classMap[boxClass], "bbox": bbox.astype(float), "iscrowd": 0, "id": (idx), "score": 1.0}

        annotations.append(tmp_anno)
        # annotations[counter] = tmp_anno
        counter += 1

# print(annotations)
# exit()

# annotations_GT = []
for key, anno in cocoGt.anns.items():
    [x1,y1,x2,y2] = anno['bbox']

    # width  = max(0, x2 - x1)
    # height = max(0, y2 - y1)
    width  = max(0, abs(x2 - x1))
    height = max(0, abs(y2 - y1))

    x1 = min(x1,x2)
    y1 = min(y1,y2)
    # print(anno['bbox'])

    anno['bbox']  = [x1, y1, width, height]
    anno['area']  = width * height
    anno['score'] = 0.99

    cocoGt.anns[key] = anno

# cocoDt = copy.deepcopy(cocoGt)
# cocoDt.anns = annotations
# print(cocoDt.anns[1])

for key, anno in cocoGt.anns.items():
    # imgId = anno['image_id']
    [x1,y1,x2,y2] = anno['bbox']


cocoEval = COCOeval(cocoGt, cocoDt, 'bbox') 

imgIds  = list(detResult.keys())
imgIds  = sorted(imgIds)
# cocoEval.params.imgIds = imgIds #参数设置

cocoEval.evaluate()    #评价
cocoEval.accumulate()  #积累
cocoEval.summarize()   #总结

exit()

'''
# APs = []
# mAp = np.mean(APs)
# print("mAp = {}".format(mAp))



cocoDt_file = 'results/coco_results.json' #需要根据自己的实际情况配置该路径
cocoDt = cocoGt.loadRes(cocoDt_file) #取得结果集中image json对象

imgIds = sorted(imgIds) #按顺序排列coco标注集image_id
imgIds = imgIds[0:5000] #标注集中的image数据

'''

'''
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

'''