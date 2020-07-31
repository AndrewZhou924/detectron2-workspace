# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

import pickle
import os, json, cv2, random
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

'''
detectron2 data format
in each record (dtype = dict):
    file_name (full path)
    image_id  0 - N
    height
    width
    annotations: list of obj
        obj: dtype = dict
            bbox [1, 4]
            bbox_mode
            segmentation
            category_id
'''

DATA_DIR    = "/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data"

def Iou(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def validateBox(boxToCheck, box_list):
    '''
        set Iou threshold to 0.75
    '''
    for box in box_list:
        if Iou(boxToCheck, box) > 0.75:
            return False
    return True

def getGTHicoTrainDataset():
    '''
        using boxes in Trainval_GT_HICO_with_pose.pkl
    '''
    dataset_dicts = []
    Trainval_GT = pickle.load( open( DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl',  "rb" ), encoding="bytes")

    for idx, img_data in tqdm(enumerate(Trainval_GT)):
        record = {}
        img_id = img_data[0]
        filename = DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(img_id)).zfill(8) + '.jpg'

        try:
            height, width = cv2.imread(filename).shape[:2]
        except:
            continue
            
        record["file_name"] = filename
        record["image_id"]  = idx
        record["height"]    = height
        record["width"]     = width
        objs = []
        obj_H = {   "bbox": img_data[2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": None,
                    "category_id": 0, }
        obj_O = {   "bbox": img_data[3],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    # "segmentation": None,
                    "category_id": 1, }
        objs.append(obj_H)
        objs.append(obj_O)
        record["annotations"] = objs

        dataset_dicts.append(record)

        # test Mode
        '''
        if idx == 10:
            break
        '''
    return dataset_dicts

def getHicoTrainDataset():
    '''
        using boxes in db_trainval.pkl
    '''
    data = pickle.load(open('./data/db_trainval.pkl','rb'))
    dataset_dicts = []

    for img_id in tqdm(data.keys()):
        record = {}

        img_data = data[img_id]
        filename = DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(img_id)).zfill(8) + '.jpg'

        try:
            height, width = cv2.imread(filename).shape[:2]
        except:
            continue
            
        record["file_name"] = filename
        record["image_id"]  = img_id
        record["height"]    = height
        record["width"]     = width
        objs = []

        for box, box_class in zip(img_data['boxes'], img_data['obj_classes']): 
            if objs != [] and validateBox(box, [obj["bbox"] for obj in objs]) == False:
                continue

            obj =   {   "bbox": box,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": box_class }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

        # test Mode
        '''
        if img_id == 20:
            break
        '''
    return dataset_dicts

DatasetCatalog.register("HICO_train", getHicoTrainDataset)
# MetadataCatalog.get("HICO_train").set(thing_classes=["balloon"])
HICO_metadata = MetadataCatalog.get("HICO_train")

# visualization of bboxes
dataset_dicts = getHicoTrainDataset()
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=HICO_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    image = out.get_image()[:, :, ::-1]
    cv2.imwrite("./output/vis-train/vis_{}".format(d["file_name"].split('/')[-1]), image)


# exit()
# train
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.configFile = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

cfg.merge_from_file(model_zoo.get_config_file(cfg.configFile))
cfg.DATASETS.TRAIN = ("HICO_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg.configFile)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.OUTPUT_DIR = "./output/"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()