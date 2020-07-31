'''
@Descripttion: 
@version: 
@Author: 周耀海 u201811260@hust.edu.cn
@LastEditTime: 2020-07-23 17:25:51
'''
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

cfg = get_cfg()
trainer = DefaultTrainer(cfg) 

register_coco_instances("coco_train_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_train.json", "./Coco/detectron2/datasets/coco/train")
register_coco_instances("coco_val_2", {}, "./Coco/detectron2/datasets/coco/annotations/instances_val.json", "./Coco/detectron2/datasets/coco/val")

evaluator = COCOEvaluator("coco_val_2",cfg,True,output_dir="./output")
val_loader = build_detection_test_loader(cfg, "coco_val_2")
print(inference_on_dataset(trainer.model, val_loader, evaluator))