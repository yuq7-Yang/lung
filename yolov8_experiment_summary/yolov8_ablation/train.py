#!/usr/bin/env python3
"""
YOLOv8损失函数消融实验训练脚本
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
                       default='/root/autodl-tmp/lung_ct/data.yaml',
                       help='数据配置文件路径')
    parser.add_argument('--output-dir', type=str, default='./runs/ablation',
                       help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    return parser.parse_args()

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
    
    return config

def main():
    args = parse_args()
    
    print(f"🔬 开始消融实验: {args.exp_name}")
    print(f"📊 损失函数: {args.exp_name.replace('_', ' ')}")
    
    # 设置输出路径
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置损失函数参数
    loss_config = get_loss_config(args.exp_name)
    
    # 训练参数（与基线模型完全一致）
    train_args = {
        'model': '/root/exp2_100epochs/weights/best.pt',  # 基线模型
        'data': args.data,
        'epochs': 100,   # 100轮训练
        'patience': 20,  # 20轮早停
        'batch': 32,     # 批次大小
        'imgsz': 640,    # 图像尺寸
        'save': True,
        'save_period': 10,
        'cache': False,
        'device': '0',   # GPU设备
        'workers': 4,
        'project': str(Path(args.output_dir).parent),
        'name': args.exp_name,
        'exist_ok': True,
        'pretrained': False,  # 使用我们的预训练模型
        'optimizer': 'SGD',
        'verbose': True,
        'seed': args.seed,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
        'format': 'torchscript',
        'keras': False,
        'optimize': False,
        'int8': False,
        'dynamic': False,
        'simplify': True,
        'opset': None,
        'workspace': 4,
        'nms': False,
        'lr0': 0.001,      # 初始学习率
        'lrf': 0.01,       # 最终学习率
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': loss_config['box'],
        'cls': loss_config['cls'],
        'dfl': loss_config['dfl'],
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'copy_paste_mode': 'flip',
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'cfg': None,
        'tracker': 'botsort.yaml',
    }
    
    # 添加Focal Loss参数
    if 'fl_gamma' in loss_config:
        train_args['fl_gamma'] = loss_config['fl_gamma']
    
    print(f"📈 训练参数配置完成")
    print(f"📁 输出目录: {output_dir}")
    print(f"⏰ 早停: 20轮无改进")
    print(f"🔄 总轮次: 100轮")
    print(f"🎲 随机种子: {args.seed}")
    
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
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
