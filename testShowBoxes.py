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
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }

            objs.append(obj)

            
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


# for d in ["train", "val"]:
#     print(lambda d=d: get_balloon_dicts("balloon/" + d))
    # DatasetCatalog.register("balloon_" + d, ))
    # MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])


# balloon_metadata = MetadataCatalog.get("balloon_train")
   
data     = pickle.load(open('./data/db_trainval.pkl','rb'))
DATA_DIR = "/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data"
Trainval_GT = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_GT_HICO_with_pose.pkl',  "rb" ), encoding="bytes")

img_id   = 1
img_meta = data[img_id]
filename = DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(img_id)).zfill(8) + '.jpg'


dataset_dicts = []

for idx, img_id in tqdm(enumerate(data.keys())):
    record = {}
    img_meta = data[img_id]
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

        obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": None,
                    "category_id": 0,
                }

        objs.append(obj)

            
    record["annotations"] = objs
    dataset_dicts.append(record)


