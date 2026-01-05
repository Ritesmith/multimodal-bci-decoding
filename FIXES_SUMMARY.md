# BCI项目修复总结

## 日期
2025年12月30日

## 已修复的问题

### 1. Events数据类型问题
**文件**: `src/data_loader.py:186-190`

**问题**:
- Events数组中部分字段为float类型
- MNE Epochs要求所有字段为整数

**修复**:
```python
events_array = np.column_stack([
    events['timestamps'],
    np.zeros(len(events['timestamps']), dtype=int),  # 改为int
    events['event_ids']
]).astype(int)  # 确保所有列为int
```

---

### 2. Events时间戳转换
**文件**: Events数据文件

**问题**:
- Events时间戳是秒数（2-835秒）
- MNE需要采样点数
- EEG采样率: 1000Hz

**修复**:
```python
# 将秒数转换为采样点
events[:, 0] = (events[:, 0] * fs).astype(int)
# 结果: 2-835秒 -> 2000-835000采样点
```

---

### 3. fNIRS时间戳同步
**文件**: `src/data_loader.py:233-239`

**问题**:
- Events时间戳是EEG采样点
- fNIRS采样率是10Hz（EEG的1/100）
- 直接使用会导致索引越界

**修复**:
```python
# 转换EEG时间戳到fNIRS时间戳
eeg_fs = 1000.0
fnirs_fs = fnirs_data['sampling_rate']
fnirs_timestamps = (events['timestamps'] / eeg_fs * fnirs_fs).astype(int)
```

---

### 4. fNIRS数据维度处理
**文件**: `src/data_loader.py:244-256`

**问题**:
- fNIRS可能是1D数组导致axis=1错误
- Baseline correction计算无效（空切片）

**修复**:
```python
# 确保数据是2D
if hbo.ndim == 1:
    hbo = hbo.reshape(1, -1)
    hbr = hbr.reshape(1, -1)

# 使用前10%数据作为baseline
baseline_samples = max(1, int(hbo_filtered.shape[1] * 0.1))
baseline_mean_hbo = np.mean(hbo_filtered[:, :baseline_samples], axis=1, keepdims=True)
```

---

### 5. Baseline模型forward方法
**文件**: `src/models.py:291, 327`

**问题**:
- `BaselineEEGModel.forward(x)` 只接受1个参数
- 训练时传入3个参数: `(x_eeg, x_fnirs, labels)`

**修复**:
```python
# EEG Baseline
def forward(self, x_eeg: torch.Tensor, x_fnirs: torch.Tensor = None) -> torch.Tensor:
    features = self.eeg_encoder(x_eeg)
    final_features = features[:, -1, :]
    logits = self.classifier(final_features)
    return logits

# fNIRS Baseline
def forward(self, x_eeg: torch.Tensor, x_fnirs: torch.Tensor) -> torch.Tensor:
    features = self.fnirs_encoder(x_fnirs)
    logits = self.classifier(features)
    return logits
```

---

### 6. 训练代码重构
**文件**: `src/experiment.py:543-630`

**问题**:
- `CrossValidationTrainer`使用lambda函数导致错误
- 无法正确展开model_params

**修复**:
- 改用`BCITrainer`直接训练
- 添加简单的train/val split
- 使用`train_test_split`替代cross-validation

---

## 数据状态

### 已加载的真实数据
- **受试者**: 3个 (subject_01, subject_02, subject_03)
- **Trials**: 357个
- **类别**: 4类 (左手、右手、脚、舌头)

### EEG数据
- **通道数**: 64
- **采样率**: 1000 Hz
- **维度**: (357, 64, 7001)
  - 357: trials
  - 64: EEG通道
  - 7001: 时间点 (7秒/epoch)

### fNIRS数据
- **通道数**: 2 (HbO + HbR)
- **采样率**: 10 Hz
- **维度**: (357, 2, 100)
  - 357: trials
  - 2: HbO和HbR
  - 100: 时间点 (10秒/epoch)

---

## 依赖安装状态

### ✅ 已安装
- Python 3.12
- PyTorch 2.9.1+cpu
- torchvision 0.24.1+cpu
- scikit-learn 1.6.1
- MNE 1.11.0
- NumPy, SciPy, Matplotlib
- TensorBoard, PyYAML

### ⚠️ 可选依赖
- optuna (超参数调优)
- plotly (可视化)

---

## 运行实验

### 命令
```bash
python main.py --mode experiment
```

### 实验流程
1. ✅ 加载真实BCI数据
2. ✅ 预处理EEG和fNIRS信号
3. 🔄 训练EEG baseline模型
4. 🔄 训练fNIRS baseline模型
5. 🔄 训练多模态融合模型
6. 🔄 评估性能指标
7. 🔄 生成报告和可视化

### 预期运行时间
- **CPU模式**: 10-20分钟
- **GPU模式**: 2-5分钟

---

## 输出结果

### 结果目录
```
results/
└── bci_multimodal_<timestamp>/
    ├── results.json      # 详细数值结果
    ├── report.md         # 人类可读报告
    ├── experiment.log     # 实验日志
    └── *.png            # 可视化图表
```

### 关键指标
- **Accuracy**: 分类准确率
- **F1-Score**: 综合性能指标
- **Precision/Recall**: 各类别性能
- **ITR**: 信息传输率 (bits/min)
- **Inference Time**: 推理延迟 (ms)

---

## 下一步

### 完整实验
运行完整实验以获得所有模型性能对比：
```bash
python main.py --mode experiment
```

### 自定义配置
创建自定义配置文件：
```yaml
# my_config.yaml
experiment_name: "my_experiment"
dataset:
  subjects: [1, 2, 3]
  test_subjects: []
training:
  epochs: 50
  learning_rate: 0.0005
  batch_size: 16
models:
  multimodal:
    enabled: true
    fusion_method: attention
```

运行自定义配置：
```bash
python main.py --mode experiment --config my_config.yaml
```

### 单独训练模式
```bash
# 只训练模型
python main.py --mode train --data-path data/raw/

# 评估训练好的模型
python main.py --mode eval --model-path models/best_model.pth --data-path data/raw/
```

---

## 技术细节

### 数据预处理管道

#### EEG
1. Notch滤波 (50Hz工频去除)
2. 带通滤波 (0.5-40Hz)
3. ICA去噪 (去除眼电/肌电伪影)
4. Epoch提取 (-0.5到2.0秒)

#### fNIRS
1. 带通滤波 (0.01-0.5Hz)
2. Baseline校正
3. Epoch提取 (考虑血流动力学延迟)

### 模型架构

#### EEG Baseline
- CNN特征提取 (2层卷积)
- LSTM时序建模 (2层)
- 分类头 (全连接层)

#### fNIRS Baseline
- FCN特征提取
- 全连接分类器

#### 多模态融合
- EEG分支: CNN + LSTM
- fNIRS分支: FCN
- 注意力融合层
- 分类器

---

## 已知限制

1. **fNIRS数据为模拟生成**
   - 真实PhysioNet主要是EEG
   - fNIRS是同步的模拟信号

2. **ICA收敛警告**
   - FastICA可能在某些情况下不收敛
   - 已添加异常处理，使用原始数据

3. **训练时间较长**
   - 使用CPU训练会比较慢
   - 建议使用GPU加速

---

## 联系与支持

如有问题，请查看：
- `USAGE_GUIDE.md` - 详细使用指南
- `README.md` - 项目介绍
- `GET_BCI_DATA.md` - 获取真实BCI数据
