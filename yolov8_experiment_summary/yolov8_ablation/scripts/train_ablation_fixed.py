#!/usr/bin/env python3
"""
YOLOv8消融实验训练脚本 - 完全使用基线参数
早停：20轮，训练：100轮
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8消融实验训练')
    parser.add_argument('--exp-name', type=str, required=True,
                       choices=['B_Focal-CIoU', 'C_EIoU', 'D_Focal-EIoU'],
                       help='实验名称')
    parser.add_argument('--data', type=str,
                       default='/root/minimal_package/datasets/data.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--params-config', type=str,
                       default='configs/baseline_params.yaml',
                       help='基线参数配置文件')
    parser.add_argument('--output-dir', type=str, default='./runs/ablation',
                       help='输出目录')
    return parser.parse_args()

def load_baseline_params(config_path):
    """加载基线训练参数"""
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    
    # 确保早停为20轮
    params['patience'] = 20
    return params

def get_loss_config(exp_name):
    """根据实验名称配置损失函数"""
    config = {
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    # 设置Focal Loss参数
    if 'Focal' in exp_name:
        config['fl_gamma'] = 1.5  # Focal Loss gamma参数
    
    # 注意：YOLOv8中EIoU需要特殊处理，默认使用CIoU
    return config

def main():
    args = parse_args()
    
    print(f"🔬 开始消融实验: {args.exp_name}")
    print(f"📊 损失函数: {args.exp_name.replace('_', ' ')}")
    
    # 加载基线参数
    print("📋 加载基线训练参数...")
    base_params = load_baseline_params(args.params_config)
    
    # 设置输出路径
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置损失函数参数
    loss_config = get_loss_config(args.exp_name)
    
    # 准备训练参数
    train_args = base_params.copy()
    
    # 更新实验特定参数
    train_args.update({
        'model': '/root/exp2_100epochs/weights/best.pt',  # 基线模型
        'data': args.data,
        'project': str(Path(args.output_dir).parent),
        'name': args.exp_name,
        'patience': 20,  # 20轮早停
        'epochs': 100,   # 100轮训练
        'box': loss_config['box'],
        'cls': loss_config['cls'],
        'dfl': loss_config['dfl'],
    })
    
    # 添加Focal Loss参数
    if 'fl_gamma' in loss_config:
        train_args['fl_gamma'] = loss_config['fl_gamma']
    
    # 移除None值参数
    train_args = {k: v for k, v in train_args.items() if v is not None}
    
    print(f"📈 训练参数配置完成")
    print(f"📁 输出目录: {output_dir}")
    print(f"⏰ 早停: 20轮无改进")
    print(f"🔄 总轮次: 100轮")
    
    # 加载模型
    print("🔧 加载基线模型...")
    model = YOLO('/root/exp2_100epochs/weights/best.pt')
    
    # 开始训练
    print("🚀 开始训练...")
    try:
        results = model.train(**train_args)
        
        # 保存实验配置
        experiment_info = {
            'experiment_name': args.exp_name,
            'base_model': '/root/exp2_100epochs/weights/best.pt',
            'data_config': args.data,
            'loss_config': loss_config,
            'training_params': train_args,
            'output_dir': str(output_dir),
            'early_stopping': 20,
            'total_epochs': 100
        }
        
        config_file = output_dir / 'experiment_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(experiment_info, f, default_flow_style=False)
        
        print(f"✅ 实验完成: {args.exp_name}")
        print(f"📝 配置文件: {config_file}")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
