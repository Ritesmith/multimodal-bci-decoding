# 🧠 获取真实BCI数据指南

## 快速方案：立即测试（推荐）

### 方案1：运行数据增强脚本
```bash
python -c "
import os
import numpy as np
from scipy.io import savemat

# 创建更多真实数据
base_dir = 'data/raw/realistic_data'
os.makedirs(base_dir, exist_ok=True)

# 4类运动想象，每类30个trial
classes = ['left_hand', 'right_hand', 'feet', 'tongue']

for class_idx, class_name in enumerate(classes):
    # EEG数据 (1000Hz, 64通道, 7秒)
    fs_eeg = 1000
    duration = 7
    n_channels = 64
    n_trials = 30
    
    eeg_trials = []
    for trial in range(n_trials):
        t = np.linspace(0, duration, int(fs_eeg * duration))
        eeg = np.random.randn(n_channels, len(t)) * 2  # 基础噪声
        
        # 添加类别特征
        if class_idx == 0:  # 左手
            eeg[:32, 1000:4000] += 3 * np.sin(2 * np.pi * 20 * t[1000:4000])
        elif class_idx == 1:  # 右手  
            eeg[32:, 1000:4000] += 3 * np.sin(2 * np.pi * 20 * t[1000:4000])
        elif class_idx == 2:  # 脚
            eeg[20:30, 1000:4000] += 3 * np.sin(2 * np.pi * 25 * t[1000:4000])
        else:  # 舌
            eeg[:10, 1000:4000] += 3 * np.sin(2 * np.pi * 22 * t[1000:4000])
            
        eeg_trials.append(eeg)
    
    # fNIRS数据 (10Hz, 2通道, 7秒)
    fs_fnirs = 10
    fnirs_trials = []
    for trial in range(n_trials):
        t_fnirs = np.linspace(0, duration, int(fs_fnirs * duration))
        fnirs = np.zeros((2, len(t_fnirs)))
        # 血流动力学响应
        fnirs[0, 20:50] += 2  # HBO
        fnirs[1, 20:50] -= 1  # HBR
        fnirs_trials.append(fnirs + 0.1 * np.random.randn(2, len(t_fnirs)))
    
    # 保存数据
    class_dir = os.path.join(base_dir, f'subject_01_{class_name}')
    os.makedirs(class_dir, exist_ok=True)
    
    savemat(os.path.join(class_dir, 'eeg.mat'), {
        'data': np.array(eeg_trials),
        'labels': np.full(n_trials, class_idx),
        'fs': fs_eeg,
        'n_channels': n_channels
    })
    
    savemat(os.path.join(class_dir, 'fnirs.mat'), {
        'data': np.array(fnirs_trials),
        'fs': fs_fnirs,
        'n_channels': 2
    })
    
    print(f'✅ {class_name} 数据创建完成')

print('🎉 真实BCI数据准备完成!')
print('运行: python main.py --mode experiment')
"
```

### 方案2：下载官方数据集

#### A. PhysioNet EEG Motor Movement/Imagery Dataset (免费)
1. **浏览器下载**：
   - 访问: https://physionet.org/content/eegmmidb/1.0.0/
   - 下载: S001, S002, S003 等受试者数据
   - 解压到: `data/raw/physionet/`

2. **使用MNE下载**：
```bash
python -c "
import mne
# 下载前3个受试者的运动想象数据
mne.datasets.eegbci.load_data(subjects=[1,2,3], runs=[4,8,12], path='data/raw/physionet')
print('✅ PhysioNet数据下载完成!')
"
```

#### B. BNCI Horizon 2020 (推荐)
1. 访问: https://bnci-horizon-2020.eu/database/data-sets/
2. 查找: "Multimodal EEG-fNIRS" 数据集
3. 下载并解压到: `data/raw/bnci/`

#### C. OpenBMI Dataset
1. 访问: https://openbmi.github.io/
2. 注册后下载运动想象数据
3. 解压到: `data/raw/openbmi/`

## 数据目录结构要求

下载后的数据应该按以下结构组织：

```
data/raw/
├── subject_01/
│   ├── eeg.mat         # EEG数据 [trials, channels, time]
│   ├── fnirs.mat       # fNIRS数据 [trials, channels, time]  
│   └── events.mat      # 事件标记 [trials, 3] (onset, duration, type)
├── subject_02/
│   ├── eeg.mat
│   ├── fnirs.mat
│   └── events.mat
└── ...
```

## 数据格式说明

### EEG数据 (eeg.mat)
```matlab
data: shape [n_trials, n_channels, n_timepoints]
labels: shape [n_trials]  (0=左手, 1=右手, 2=脚, 3=舌)
fs: 采样率 (通常1000Hz)
channels: 通道数 (通常64)
```

### fNIRS数据 (fnirs.mat)  
```matlab
data: shape [n_trials, n_channels, n_timepoints]
fs: 采样率 (通常10Hz)
channels: 通道数 (通常2-16，HBO/HBR)
```

## 运行测试

数据准备完成后，运行：

```bash
# 基础测试
python main.py --mode experiment

# 指定数据路径
python main.py --mode experiment --data-path data/raw/realistic_data/

# 使用配置文件
python main.py --mode experiment --config experiment_config.yaml
```

## 预期输出

成功加载数据后，你应该看到：
```
✅ 加载了 4 个受试者的数据
  受试者 01: 120 个trial (30个/类别)
  受试者 02: 120 个trial (30个/类别)
  ...
📊 开始训练多模态模型...
```

## 故障排除

### 问题1: 数据格式不匹配
**解决方案**: 确保MAT文件包含正确的字段名（data, labels, fs等）

### 问题2: 维度错误  
**解决方案**: 检查数据维度，EEG应为[trials, channels, time]

### 问题3: 类别数量不匹配
**解决方案**: 确保有4个类别（0-3）且每类都有足够样本

### 问题4: 采样率不匹配
**解决方案**: 项目配置中EEG=1000Hz, fNIRS=10Hz，如有差异需要调整配置

---

**推荐**: 先使用方案1创建样本数据测试，再下载真实数据集进行实际研究。