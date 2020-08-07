from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

register_coco_instances("HICO-det-train", {}, "./data/HICO-DET-Detector/hico_annotations_train2015.json", "/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data/hico_20160224_det/images/train2015")
register_coco_instances("HICO-det-test",  {}, "./data/HICO-DET-Detector/hico_annotations_test2015.json",  "/disk1/zhanke/TIN/Transferable-Interactiveness-Network/Data/hico_20160224_det/images/test2015")

cfg = get_cfg()
cfg.configFile     = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# cfg.MODEL.WEIGHTS  = "./output/model_final.pth"
cfg.MODEL.WEIGHTS  = "./data/HICO-DET-Detector/model_0064999.pth" # VCL model map = 30.79%
cfg.DATASETS.TRAIN = ("HICO-det-train")
cfg.DATASETS.TEST  = ("HICO-det-test")

trainer    = DefaultTrainer(cfg) 
evaluator  = COCOEvaluator("HICO-det-test", cfg, True, output_dir="./output")
val_loader = build_detection_test_loader(cfg, "HICO-det-test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))