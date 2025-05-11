from ultralytics import YOLO
import os
import torch.multiprocessing as mp


def validate_model(model_path, data_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载模型
    model = YOLO(model_path)

    # 验证模型
    metrics = model.val(data=data_path)  # 需要指定数据集配置文件路径
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP75: {metrics.box.map75:.4f}")
    print(f"mAP50-95 per category: {metrics.box.maps}")


if __name__ == '__main__':
    # 在Windows上运行多进程时需要这个保护
    mp.freeze_support()

    # 数据集配置文件路径
    data_path = r'C:\Users\1\Downloads\yolo\ultralytics-main\ultralytics\cfg\datasets\coco.yaml'

    # 验证官方模型
    validate_model('yolov8n.pt', data_path)

    # 验证自定义模型
    validate_model(r'C:\Users\1\Downloads\yolo\runs\detect\train\weights\best.pt', data_path)
