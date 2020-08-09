CUDA_VISIBLE_DEVICES=1 ./train_net.py \
--config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
--eval-only     MODEL.WEIGHTS /home/zhanke/workspace/detectron2/detectron2-workspace/data/HICO-DET-Detector/model_0064999.pth
