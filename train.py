from ultralytics import YOLO
import os
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./yolov8n-ab.yaml', help='model.yaml path')
    parser.add_argument('--name', type=str, default='a')
    return parser.parse_args()

def train_model():
    args = parse_opt()
    model = YOLO(args.cfg)
    model.train(data='./coco.yaml', epochs=200, imgsz=640, batch=16, project='./exps', name=args.name,optimizer="SGD")

if __name__ == '__main__':
    train_model()

#YOLOv8n summary: 225 layers, 3011433 parameters, 3011417 gradients, 8.2 GFLOPs