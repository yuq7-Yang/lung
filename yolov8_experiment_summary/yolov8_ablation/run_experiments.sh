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
DATA_CONFIG="/root/autodl-tmp/lung_ct/data.yaml"
OUTPUT_DIR="./runs/ablation_study"
TRAIN_SCRIPT="train.py"

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
        echo "请检查路径是否正确"
        exit 1
    fi
    
    # 检查数据配置
    if [ ! -f "$DATA_CONFIG" ]; then
        echo "❌ 错误: 数据配置文件不存在: $DATA_CONFIG"
        echo "请检查路径是否正确"
        exit 1
    fi
    
    # 检查训练脚本
    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "❌ 错误: 训练脚本不存在: $TRAIN_SCRIPT"
        exit 1
    fi
    
    # 设置不同的随机种子
    local seed=$((42 + RANDOM % 1000))
    
    # 执行训练
    echo "🎲 使用随机种子: $seed"
    python "$TRAIN_SCRIPT" \
        --exp-name "$exp_name" \
        --data "$DATA_CONFIG" \
        --output-dir "$OUTPUT_DIR" \
        --seed "$seed" \
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
echo "  训练脚本: $TRAIN_SCRIPT"
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
        echo "⏸️  实验间暂停 30秒..."
        sleep 30
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
echo ""
echo "实验汇总报告:"
echo "=================================================="
for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "实验: $exp"
    exp_dir="$OUTPUT_DIR/$exp"
    
    if [ -d "$exp_dir" ]; then
        echo "  输出目录: $exp_dir"
        echo "  早停: 20轮"
        echo "  总轮次: 100轮"
        
        # 检查是否有最佳模型
        if [ -f "$exp_dir/weights/best.pt" ]; then
            echo "  ✓ 最佳模型: $exp_dir/weights/best.pt"
        else
            echo "  ⚠️  最佳模型未找到"
        fi
        
        # 检查配置文件
        if [ -f "$exp_dir/experiment_config.yaml" ]; then
            echo "  ✓ 配置文件: $exp_dir/experiment_config.yaml"
        fi
    else
        echo "  ⚠️  实验目录未找到"
    fi
done
echo "=================================================="
