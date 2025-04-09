from ultralytics import YOLO

# 加载一个预训练的 YOLO11n 模型
model = YOLO("/workspace/Outputs/yyh/yolo11/experiment250409/train4/weights/best.pt")
# model.load('/workspace/Outputs/yyh/yolo11/experiment250409/train4/weights/best.pt')  # loading pretrain weights

model.val(
    data="/workspace/mtmct/ContionTrack/fork/ultralytics/ultralytics/cfg/datasets/hit-uav.yaml",  # 数据集配置文件路径
    # epochs=100,  # 训练周期数
    imgsz=640,  # 训练图像尺寸
    device=0,  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
    project="/workspace/Outputs/yyh/yolo11/experiment250409",  # 项目名称
    amp=False,  # 是否使用自动混合精度
    cache=False,  # 是否使用缓存
)