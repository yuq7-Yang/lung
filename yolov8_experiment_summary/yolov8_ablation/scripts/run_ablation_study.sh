#!/bin/bash

# ============================================
# YOLOv8损失函数消融实验批量执行脚本
# 基线模型: /root/exp2_100epochs/weights/best.pt
# 早停: 20轮，训练: 100轮
# ============================================

set -e  # 遇到错误退出

echo "=========================================="
echo "🚀 YOLOv8损失函数消融实验启动"
echo "📅 时间: $(date)"
echo "=========================================="

# 设置环境
export PYTHONPATH=/usr/local/lib/python3.8/dist-packages
export CUDA_VISIBLE_DEVICES=0

# 基础路径
BASE_MODEL="/root/exp2_100epochs/weights/best.pt"
DATA_CONFIG="/root/minimal_package/datasets/data.yaml"
PARAMS_CONFIG="configs/baseline_params.yaml"
OUTPUT_DIR="./runs/ablation_study"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p logs

# 实验列表（按顺序执行）
EXPERIMENTS=(
    "B_Focal-CIoU"
    "C_EIoU"
    "D_Focal-EIoU"
)

# 记录开始时间
START_TIME=$(date +%s)

# 函数：检查GPU状态
check_gpu_status() {
    echo "🖥️  检查GPU状态..."
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv
    echo ""
}

# 函数：运行单个实验
run_experiment() {
    local exp_name=$1
    local log_file="logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "=========================================="
    echo "🔬 启动实验: $exp_name"
    echo "📝 日志文件: $log_file"
    echo "=========================================="
    
    # 检查基线模型是否存在
    if [ ! -f "$BASE_MODEL" ]; then
        echo "❌ 错误: 基线模型不存在: $BASE_MODEL"
        exit 1
    fi
    
    # 检查数据配置
    if [ ! -f "$DATA_CONFIG" ]; then
        echo "❌ 错误: 数据配置文件不存在: $DATA_CONFIG"
        exit 1
    fi
    
    # 检查参数配置
    if [ ! -f "$PARAMS_CONFIG" ]; then
        echo "❌ 错误: 参数配置文件不存在: $PARAMS_CONFIG"
        exit 1
    fi
    
    # 执行训练
    python train_ablation_fixed.py \
        --exp-name "$exp_name" \
        --data "$DATA_CONFIG" \
        --params-config "$PARAMS_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 实验完成: $exp_name"
        echo "📊 结果保存在: $OUTPUT_DIR/$exp_name"
    else
        echo "❌ 实验失败: $exp_name (退出码: $exit_code)"
        echo "📋 查看日志: $log_file"
    fi
    
    echo ""
}

# 主执行流程
echo "🔍 检查环境..."
check_gpu_status

echo "📋 实验配置:"
echo "  基线模型: $BASE_MODEL"
echo "  数据配置: $DATA_CONFIG"
echo "  参数配置: $PARAMS_CONFIG"
echo "  输出目录: $OUTPUT_DIR"
echo "  早停轮次: 20"
echo "  训练轮次: 100"
echo ""

echo "📌 实验队列: ${#EXPERIMENTS[@]} 个实验"
for i in "${!EXPERIMENTS[@]}"; do
    echo "  $((i+1)). ${EXPERIMENTS[$i]}"
done
echo ""

# 逐个执行实验
for exp in "${EXPERIMENTS[@]}"; do
    run_experiment "$exp"
    
    # 实验间短暂暂停
    if [ "$exp" != "${EXPERIMENTS[-1]}" ]; then
        echo "⏸️  实验间暂停 10秒..."
        sleep 10
        check_gpu_status
    fi
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo "=========================================="
echo "🎉 所有实验完成!"
echo "⏱️  总耗时: $((TOTAL_TIME / 3600))小时 $(( (TOTAL_TIME % 3600) / 60 ))分钟 $((TOTAL_TIME % 60))秒"
echo "📁 结果目录: $OUTPUT_DIR"
echo "📋 实验摘要:"
echo "  1. B_Focal-CIoU - Focal-CIoU损失"
echo "  2. C_EIoU - EIoU损失" 
echo "  3. D_Focal-EIoU - Focal-EIoU损失"
echo "=========================================="

# 生成汇总报告
echo "📊 生成实验汇总..."
python -c "
import yaml
import glob
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
experiments = ['B_Focal-CIoU', 'C_EIoU', 'D_Focal-EIoU']

print('实验汇总报告:')
print('=' * 50)
for exp in experiments:
    config_file = output_dir / exp / 'experiment_config.yaml'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f'\\n实验: {exp}')
        print(f'  输出目录: {config[\"output_dir\"]}')
        print(f'  损失配置: {config[\"loss_config\"]}')
        print(f'  早停: {config[\"early_stopping\"]}轮')
        print(f'  总轮次: {config[\"total_epochs\"]}')
    else:
        print(f'\\n⚠️  实验配置未找到: {exp}')
print('=' * 50)
"
