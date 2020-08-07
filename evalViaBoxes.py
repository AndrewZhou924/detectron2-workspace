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

trainer     = DefaultTrainer(cfg) 
evaluator   = COCOEvaluator("HICO-det-test", cfg, True, output_dir="./output")
evaluator.reset()

'''
build input and outputs
inputs – the inputs to a COCO model (e.g., GeneralizedRCNN). 
    It is a list of dict. Each dict corresponds to an image and 
    contains keys like “height”, “width”, “file_name”, “image_id”.

outputs – the outputs of a COCO model. 
    It is a list of dicts with key “instances” that contains Instances.
'''

# hicoTestResult

for inputs, outputs in get_all_inputs_outputs():
  evaluator.process(inputs, outputs)

eval_results = evaluator.evaluate()