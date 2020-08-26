CUDA_VISIBLE_DEVICES=0 ./train_net.py --config-file ../configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml \
--resume MODEL.WEIGHTS ./output/model_0054999.pth \
SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.001