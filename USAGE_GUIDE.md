# 使用指南

本指南将详细介绍如何使用多模态脑机接口运动意图解码系统。

## 快速开始

### 1. 环境准备

确保您的系统满足以下要求：
- Python 3.8或更高版本
- 8GB以上内存
- CUDA支持的GPU（可选，用于加速训练）

安装项目依赖：

```bash
# 克隆项目
git clone <repository-url>
cd 脑机接口课程遐想

# 创建虚拟环境
python -m venv bci_env
source bci_env/bin/activate  # Linux/Mac
# 或 bci_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

#### 支持的数据集

1. **BNCI Horizon 2020 Dataset** (推荐)
   - 包含40名受试者的同步EEG-fNIRS数据
   - 4类运动想象任务
   - 下载地址: [BNCI Website](https://bnci-horizon-2020.eu/)

2. **OpenBMI Dataset**
   - 多模态信号（EEG+fNIRS+眼动仪）
   - 适合验证跨模态融合效果
   - 下载地址: [OpenBMI Project](https://openbmi.github.io/)

#### 数据组织结构

将下载的数据按照以下结构组织：

```
data/raw/
├── bnci/
│   ├── subject_01/
│   │   ├── eeg.mat
│   │   ├── fnirs.mat
│   │   └── events.mat
│   ├── subject_02/
│   │   └── ...
│   └── ...
└── openbmi/
    ├── session_01/
    │   └── ...
    └── ...
```

### 3. 运行第一个实验

使用默认配置运行完整实验：

```bash
python main.py --mode experiment
```

这将：
1. 自动创建配置文件 `experiment_config.yaml`
2. 加载并预处理数据
3. 训练基线模型和多模态模型
4. 生成评估报告和可视化结果

结果将保存在 `results/` 目录下。

## 详细使用说明

### 配置文件详解

#### 创建自定义配置

```yaml
# experiment_config.yaml
experiment_name: "custom_bci_experiment"

dataset:
  type: "bnci"                    # 数据集类型: "bnci" 或 "openbmi"
  path: "data/raw"               # 数据路径
  subjects: [1, 2, 3, 4, 5]      # 训练受试者列表
  test_subjects: [6, 7]          # 测试受试者列表

models:
  multimodal:
    enabled: true
    fusion_method: "attention"   # 融合方法: "attention", "concat", "weighted"
    eeg_stream:
      conv_filters: [32, 64, 128]
      lstm_units: [128, 64]
      dropout: 0.3
    fnirs_stream:
      dense_units: [128, 64]
      dropout: 0.2
    attention_units: 64
  
  eeg_baseline:
    enabled: true
  
  fnirs_baseline:
    enabled: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping: true
  patience: 10
  optimizer: "AdamW"
  weight_decay: 0.0001

evaluation:
  cv_folds: 5
  realtime_threshold_ms: 200

hyperparameter_tuning:
  enabled: false
  method: "optuna"               # "grid", "random", "optuna"
  n_trials: 100
  timeout_hours: 24
```

### 运行模式详解

#### 1. 实验模式 (推荐)

完整的实验流程，包括：
- 数据加载和预处理
- 基线模型训练和评估
- 多模态模型训练和评估
- 超参数调优（可选）
- 结果分析和可视化

```bash
# 使用默认配置
python main.py --mode experiment

# 使用自定义配置
python main.py --mode experiment --config my_config.yaml --data-path /path/to/data

# 只运行特定受试者
python main.py --mode experiment --config config_with_subjects.yaml
```

#### 2. 训练模式

单独训练模型，适合调试和快速测试：

```bash
python main.py --mode train --data-path data/raw/
```

训练进度将实时显示，包括：
- 损失函数变化
- 训练/验证准确率
- 预计剩余时间

最佳模型会自动保存为 `best_model.pth`

#### 3. 评估模式

评估预训练模型的性能：

```bash
python main.py --mode eval --model-path models/best_model.pth --data-path data/raw/
```

评估包括：
- 分类指标（准确率、精确率、召回率、F1分数）
- 信息传输率(ITR)
- 实时性能（推理延迟）
- 可视化图表（混淆矩阵、决策边界等）

#### 4. 实时模式

测试实时推理系统：

```bash
python main.py --mode realtime --model-path models/best_model.pth
```

系统将：
- 启动数据模拟器
- 实时处理和解码
- 显示预测结果和性能统计
- 按Ctrl+C停止

## 高级功能

### 超参数调优

#### 网格搜索

```yaml
hyperparameter_tuning:
  enabled: true
  method: "grid"
  parameters:
    learning_rate: [0.0001, 0.001, 0.01]
    batch_size: [16, 32, 64]
    weight_decay: [0.00001, 0.0001, 0.001]
```

#### Optuna优化

```yaml
hyperparameter_tuning:
  enabled: true
  method: "optuna"
  n_trials: 100
  timeout_hours: 24
```

### 自定义模型

继承基础类创建自定义模型：

```python
from src.models import MultiModalFusionModel
import torch.nn as nn

class CustomFusionModel(MultiModalFusionModel):
    def __init__(self, config=None):
        super().__init__(config)
        # 自定义修改
        
    def forward(self, eeg_input, fnirs_input):
        # 自定义前向传播
        pass
```

### 特征工程扩展

```python
from src.feature_extraction import EEGFeatureExtractor

class CustomEEGExtractor(EEGFeatureExtractor):
    def extract_custom_features(self, eeg_epochs):
        # 实现自定义特征提取
        pass
```

## 性能优化建议

### 1. 数据预处理优化

- 使用SSD存储数据
- 预处理数据并保存到磁盘
- 使用多进程并行处理

### 2. 模型训练优化

- 启用混合精度训练
- 使用梯度累积处理大批次
- 调整数据加载器工作进程数

### 3. 实时推理优化

- 模型量化和剪枝
- 使用TensorRT加速
- 部署到边缘设备

## 故障排除

### 常见问题

#### 1. 内存不足

```bash
# 减小批次大小
python main.py --mode experiment --config config_small_batch.yaml
```

```yaml
# config_small_batch.yaml
training:
  batch_size: 16  # 减小批次大小
```

#### 2. CUDA内存错误

```python
# 在config.py中设置
DEVICE = "cpu"  # 强制使用CPU
```

#### 3. 数据加载失败

检查：
- 数据路径是否正确
- 数据文件是否完整
- 文件格式是否受支持

#### 4. 训练过慢

- 检查GPU是否被正确使用
- 减少网络复杂度
- 使用更少的数据进行测试

### 日志分析

日志文件位置：`logs/bci_project.log`

常见日志信息：
- `INFO`: 正常运行信息
- `WARNING`: 警告信息，可能影响性能
- `ERROR`: 错误信息，需要处理

```bash
# 查看最新日志
tail -f logs/bci_project.log

# 搜索错误
grep ERROR logs/bci_project.log
```

## 结果分析

### 实验报告解读

实验完成后，在 `results/<experiment_name>/` 目录下会生成：

1. **results.json**: 详细数值结果
2. **report.md**: 人类可读的实验报告
3. **experiment.log**: 实验日志
4. **可视化图表**: 各种分析图表

### 关键指标解读

- **准确率 > 80%**: 良好的分类性能
- **ITR > 15 bits/min**: 实用通信速率
- **推理时间 < 200ms**: 满足实时要求
- **F1分数**: 综合性能指标

### 可视化图表说明

1. **混淆矩阵**: 显示各类别分类情况
2. **训练曲线**: 监控训练过程
3. **特征重要性**: 识别关键脑区/通道
4. **决策边界**: 理解模型决策过程

## 扩展开发

### 添加新的数据集

1. 在 `src/data_loader.py` 中添加新的加载器类
2. 实现必要的数据加载方法
3. 更新工厂函数 `create_dataset_loader`

### 添加新的评估指标

1. 在 `src/training.py` 中的 `evaluate_model_performance` 函数中添加
2. 在 `src/evaluation.py` 中添加可视化方法
3. 更新配置文件中的评估指标列表

### 添加新的融合方法

1. 在 `src/feature_extraction.py` 中实现新的融合策略
2. 在 `src/models.py` 中更新融合层
3. 在配置文件中添加新方法选项

## 最佳实践

### 1. 实验设计

- 始终使用交叉验证
- 设置随机种子确保可重复性
- 记录所有实验参数和结果

### 2. 代码管理

- 使用版本控制
- 编写清晰的文档字符串
- 添加适当的单元测试

### 3. 性能监控

- 监控GPU内存使用
- 记录训练时间
- 定期检查模型过拟合

### 4. 结果复现

- 保存完整的配置文件
- 记录数据版本
- 提供详细的环境信息

---

如有更多问题，请参考项目的GitHub Issues或联系开发团队。