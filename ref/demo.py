import torch, torchvision
import os
import shutil
# print(torch.__version__, torch.cuda.is_available())
print("Demo version. Output pictures are stored in ./demo dir")

demo_output_dir="./demo"
if not os.path.exists(demo_output_dir):
    os.makedirs(demo_output_dir)


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog

import numpy as np
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
register_coco_instances("coco_train_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_train.json", "./Coco/detectron2/datasets/coco/train")
register_coco_instances("coco_val_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_val.json", "./Coco/detectron2/datasets/coco/val")



from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("coco_train_2",)
cfg.DATASETS.TEST = ("coco_val_2", )
cfg.DATALOADER.NUM_WORKERS = 2


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)#128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # how many classes
cfg.MODEL.RPN.NMS_THRESH = 0.75
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 96, 108, 128]]
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model

predictor = DefaultPredictor(cfg)


img_path = "./Coco/detectron2/datasets/coco/val/"
files = os.listdir("./Coco/detectron2/datasets/coco/val")
i = 1

#load some pics from image_val to test
for file in files:
    if i > 10:
        break
    img = os.path.join(img_path,file)
    im = cv2.imread(img)
    outputs = predictor(im)
    print(i)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)#为什么这里是cfg.DATASETS.TRAIN[0]而不是TEST呢？
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out.save("./demo/model_output"+format(str(i))+".jpg")
    i += 1

# #from detectron2.utils.visualizer import ColorMode
# dataset_dicts = load_coco_json("/home/zyh/detectron2_1/Coco/detectron2/datasets/coco/annotations/instances_val.json", "/home/zyh/detectron2_1/Coco/detectron2/datasets/coco/val")
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
#                    scale=0.8, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     # cv2_imshow(out.get_image()[:, :, ::-1])
#     out.save("./demo/model_output.jpg")

