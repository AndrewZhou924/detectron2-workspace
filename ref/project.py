import torch, torchvision
# import some common libraries
import os
import shutil
import numpy as np
import cv2
import random
import numpy as np
import json
import random

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json

'''
    code for main project
'''

setup_logger()

output_dir= "./output/"
if os.path.exists(output_dir):
	shutil.rmtree(output_dir)

register_coco_instances("coco_train_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_train.json", "./Coco/detectron2/datasets/coco/train")
register_coco_instances("coco_val_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_val.json", "./Coco/detectron2/datasets/coco/val")

# from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import builtin,load_coco_json
dataset_dicts = load_coco_json("./Coco/detectron2/datasets/coco/annotations/instances_train.json", "./Coco/detectron2/datasets/coco/train")
# print(dataset_dicts)

coco_train_metadata=MetadataCatalog.get("coco_train_2")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=coco_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    # cv2_imshow(vis.get_image()[:, :, ::-1])
    vis.save("./output.jpg")


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
# print(os.getcwd())
#os.chdir(r"/home/xuyifei/anaconda3/envs/detectron2/lib/python3.7/site-packages/detectron2")
# print(os.getcwd())
cfg = get_cfg()
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
# https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml


cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.DATASETS.TRAIN = ("coco_train_2",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2 #2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)#128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # how many classes
cfg.MODEL.RPN.NMS_THRESH = 0.75
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.53, 0.56, 0.63]]# [[0.5, 1.0, 2.0]]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[98, 143, 192, 238, 268]]# [[32, 64, 96, 108, 128]]



os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg.OUTPUT_DIR)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



from detectron2.utils.visualizer import ColorMode

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("coco_val_2", )
predictor = DefaultPredictor(cfg)


img_path = "./Coco/detectron2/datasets/coco/val/"
files = os.listdir("./Coco/detectron2/datasets/coco/val")
i = 0 

#load some pics from image_val to test
for file in files:
    if i > 3:
        break
    img = os.path.join(img_path,file)
    im = cv2.imread(img)
    outputs = predictor(im)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)#为什么这里是cfg.DATASETS.TRAIN[0]而不是TEST呢？
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])
    out.save("./model_output.jpg")
    i += 1

# #from detectron2.utils.visualizer import ColorMode
# dataset_dicts = load_coco_json("/home/xuyifei/anaconda3/envs/detectron2/lib/python3.7/site-packages/detectron2/datasets/coco/annotations/instances_val.json", "/home/xuyifei/anaconda3/envs/detectron2/lib/python3.7/site-packages/detectron2/datasets/coco/val")
# for d in random.sample(dataset_dicts, 3):    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=coco_train_metadata, 
#                    scale=0.8, 
#                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
#     )
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2_imshow(out.get_image()[:, :, ::-1])


print(os.getcwd())
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


evaluator = COCOEvaluator("coco_val_2",cfg,True,output_dir="./output")
val_loader = build_detection_test_loader(cfg, "coco_val_2")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
# another equivalent way is to use trainer.test


