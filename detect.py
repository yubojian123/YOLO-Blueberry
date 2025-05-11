import warnings

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs/detect/train/weights/best.pt') # select your model.pt path
    model.predict(source=r'C:\Users\1\Downloads\yolo\ultralytics-main\data\my_dataset\test\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )