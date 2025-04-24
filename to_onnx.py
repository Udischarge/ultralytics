from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

# 加载一个预训练的 YOLO11n 模型
model = YOLO("/workspace/Outputs/yyh/yolo11/experiment250421/train2/weights/best.pt")

path = model.export(format='onnx', simplify=True)


# 输入和输出路径
# quantized_model_path = os.path.join(path.split('/')[:-1], "model_quantized.onnx")
quantized_model_path = "/workspace/Outputs/yyh/yolo11/onnxruntime/model_quantized.onnx"

# 动态量化
quantize_dynamic(
    model_input=path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # 将权重量化为 int8
)

print(f"Quantized model saved to: {quantized_model_path}")