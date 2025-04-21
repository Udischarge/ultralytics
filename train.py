from ultralytics import YOLO, RTDETR

# 加载一个预训练的 YOLO11n 模型
# model = RTDETR("/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml")
model = YOLO("/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/models/11/yolo11s.yaml")
model.load('/workspace/Weights/YOLO11/yolo11s.pt')  # loading pretrain weights

train_results = model.train(

    # data="/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/custom/hit-uav.yaml",  # 数据集配置文件路径
    # data="/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/custom/drone_vehicle.yaml",  # 数据集配置文件路径
    data="/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/custom/vtmot_wurenji0302.yaml",  # 数据集配置文件路径
    epochs=20,  # 训练周期数
    imgsz=640,  # 训练图像尺寸
    device=[0, 1, 2, 3],  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
    project="/workspace/Outputs/yyh/yolo11/experiment250421",  # 项目名称
    amp=False,  # 是否使用自动混合精度
    cache=False,  # 是否使用缓存
    batch=32,  # 批大小
)