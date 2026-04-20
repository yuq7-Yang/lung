# YOLOv8消融实验汇总

## 实验信息
- 基线模型: /root/exp2_100epochs/weights/best.pt
- 数据配置: /root/minimal_package/datasets/data.yaml
- 训练参数: 100 epochs, batch=32, imgsz=640, SGD(lr=0.001), patience=20

## 实验结果
| 实验 | 最佳mAP50 | 最佳mAP50-95 | 相对基线改进 |
|------|-----------|--------------|--------------|
| 基线 | 0.8855    | 0.5404       | -            |
| Focal-CIoU | 0.8829 | 0.5303 | -0.29% |
| EIoU | 0.8829 | 0.5303 | -0.29% |
| Focal-EIoU | 0.8829 | 0.5303 | -0.29% |

## 关键发现
1. 所有消融实验性能非常接近
2. 均略低于基线模型
3. 早停有效（均在20轮左右停止）

## 文件列表
total 516
drwxr-xr-x 3 root root   4096 Feb 10 13:28 .
drwx------ 1 root root   4096 Feb 10 13:28 ..
-rw-r--r-- 1 root root 427123 Feb 10 13:28 baseline_vs_ablation_comparison.png
-rw-r--r-- 1 root root    683 Feb 10 13:28 complete_comparison.csv
-rw-r--r-- 1 root root   1248 Feb 10 13:28 experiment_report.txt
-rw-r--r-- 1 root root    424 Feb 10 13:28 final_results.csv
-rw-r--r-- 1 root root    318 Feb 10 13:28 improvement_vs_baseline.csv
-rw-r--r-- 1 root root    681 Feb 10 13:28 README.md
-rw-r--r-- 1 root root  69630 Feb 10 13:28 results_comparison.png
drwxr-xr-x 6 root root    149 Feb 10 13:28 yolov8_ablation
