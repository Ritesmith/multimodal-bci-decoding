# 多模态BCI项目安装指南

## 问题诊断

您的系统出现了 `pip` 模块不可用的问题。这通常是由于：
1. Python安装时没有包含pip
2. 环境变量配置问题
3. 使用了不完整的Python安装

## 🚀 快速解决方案

### 方案1: 使用修复脚本

```bash
python quick_fix.py
```

这个脚本会：
- 自动检测可用的Python和pip
- 尝试多种安装方法
- 提供详细的修复建议

### 方案2: 重新安装pip

1. **下载pip安装器**:
   ```bash
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   ```
   
2. **安装pip**:
   ```bash
   python get-pip.py
   ```

3. **安装项目依赖**:
   ```bash
   pip install -r requirements.txt
   ```

### 方案3: 使用Conda (推荐)

1. **安装Miniconda**: https://docs.conda.io/en/latest/miniconda.html

2. **创建专用环境**:
   ```bash
   conda create -n bci python=3.12
   conda activate bci
   ```

3. **安装依赖**:
   ```bash
   conda install numpy matplotlib scipy scikit-learn pandas tqdm
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install mne pywt seaborn plotly optuna pyyaml h5py
   ```

4. **运行项目**:
   ```bash
   python main.py --mode experiment
   ```

### 方案4: 重新安装Python

1. 从 https://python.org 下载Python 3.12
2. 安装时勾选：
   - ✅ Add Python to PATH
   - ✅ Install pip
   - ✅ Install for all users
3. 重启电脑
4. 运行安装脚本

## 🧪 测试修复

安装完成后，测试修复效果：

```bash
python test_fix.py
```

这会验证：
- ✅ 基础模块导入
- ✅ 安全导入机制  
- ✅ 错误处理功能
- ✅ 主程序可用性

## 🎯 运行项目

```bash
python main.py --mode experiment
```

现在程序会：
- ✅ 友好处理缺失依赖
- ✅ 提供清晰的错误信息
- ✅ 指导安装缺失的包

## 📦 最小依赖方案

如果完整安装遇到问题，可以先安装最小依赖：

```bash
pip install numpy matplotlib scipy
```

这样至少可以运行基础功能并看到友好的错误提示。

## 🔧 常见问题

### Q: pip命令不存在
**A**: 使用 `python -m pip install <package>` 或重新安装Python

### Q: 某些包安装失败
**A**: 
- 使用国内镜像: `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>`
- 使用管理员权限运行
- 使用conda替代

### Q: 运行时仍有ImportError
**A**: 这是正常的！修复后的代码会：
- 显示具体缺少哪些包
- 提供安装命令
- 继续运行基础功能

## 💡 推荐工作流程

1. **运行快速修复**: `python quick_fix.py`
2. **测试效果**: `python test_fix.py`  
3. **运行项目**: `python main.py --mode experiment`
4. **根据提示安装缺失依赖**

## 📞 获取帮助

如果仍有问题，可以：
1. 查看 `test_fix.py` 的详细输出
2. 检查 Python 和 pip 版本
3. 尝试使用虚拟环境

---

**修复后的项目特点**:
- 🛡️ 安全导入机制 - 优雅处理依赖缺失
- 📋 清晰错误提示 - 指导用户解决问题  
- 🎯 渐进式安装 - 支持部分依赖运行
- 🔧 自动检测 - 智能选择安装方法