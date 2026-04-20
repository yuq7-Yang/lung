#!/usr/bin/env python3
"""
测试脚本 - 验证环境
"""

import sys
import torch
from ultralytics import YOLO

print("🧪 测试YOLOv8环境...")
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")

# 测试YOLO导入
try:
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8导入成功")
except Exception as e:
    print(f"❌ YOLOv8导入失败: {e}")

# 检查基线模型
import os
baseline_path = "/root/exp2_100epochs/weights/best.pt"
if os.path.exists(baseline_path):
    print(f"✅ 基线模型存在: {baseline_path}")
    print(f"文件大小: {os.path.getsize(baseline_path) / 1024 / 1024:.2f} MB")
else:
    print(f"❌ 基线模型不存在: {baseline_path}")

# 检查数据配置
data_path = "/root/autodl-tmp/lung_ct/data.yaml"
if os.path.exists(data_path):
    print(f"✅ 数据配置文件存在: {data_path}")
else:
    print(f"❌ 数据配置文件不存在: {data_path}")
