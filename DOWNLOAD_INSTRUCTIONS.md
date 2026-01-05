# 🚀 紧急修复指南 - pip完全缺失

## 问题分析
您的系统完全没有pip和conda，需要手动安装pip。

## 📥 第一步：下载pip安装器

### 方法1：浏览器下载（推荐）
1. 打开浏览器
2. 访问：https://bootstrap.pypa.io/get-pip.py
3. 右键 -> 另存为 -> 保存到：`D:\Stazica\my_files\University_works\self study\脑机接口课程遐想\get-pip.py`

### 方法2：运行下载脚本
```bash
python manual_install.py
```

## 🔧 第二步：安装pip

在项目目录下打开命令提示符，运行：

```bash
python get-pip.py
```

如果成功，您会看到类似输出：
```
Successfully installed pip-24.0 wheel-0.43.0
```

## 📦 第三步：安装基础依赖

```bash
python -m pip install numpy matplotlib scipy
```

## 🧪 第四步：测试修复

```bash
python test_fix.py
```

## 🎯 第五步：运行项目

```bash
python main.py --mode experiment
```

---

## 🆘 如果仍然失败

### 方案A：使用PowerShell（管理员权限）
1. 右键点击PowerShell -> "以管理员身份运行"
2. 导航到项目目录
3. 重复上述步骤

### 方案B：重新安装Python
1. 卸载当前Python
2. 从 https://python.org 下载Python 3.12
3. 安装时必须勾选：
   - ✅ Add Python to PATH
   - ✅ Install pip  
   - ✅ Install for all users
4. 重启电脑
5. 重新运行上述步骤

### 方案C：使用在线Python环境（临时）
如果急需测试，可以使用在线Python环境：
- https://replit.com/
- https://colab.research.google.com/

## 📞 快速验证

运行以下命令检查是否成功：

```bash
# 检查pip
python -m pip --version

# 检查numpy
python -c "import numpy; print('numpy OK')"

# 检查matplotlib  
python -c "import matplotlib; print('matplotlib OK')"
```

## 💡 重要提醒

1. **必须重启命令行** - 安装pip后需要新开命令行窗口
2. **使用完整路径** - 如果pip不在PATH，使用 `python -m pip`
3. **网络问题** - 如果下载失败，可能需要代理或VPN
4. **权限问题** - 使用管理员权限运行命令行

## 🎯 预期结果

安装成功后，运行 `python main.py --mode experiment` 应该看到：
- ✅ 友好的错误提示（而不是崩溃）
- 📋 具体缺失的依赖列表
- 💡 清晰的安装指导

---

## 🔍 故障排除

### 错误：`No module named 'pip'`
**解决方案**：确保 `get-pip.py` 下载完整，然后重新运行

### 错误：`SSL certificate verification failed`
**解决方案**：
```bash
python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org numpy
```

### 错误：权限被拒绝
**解决方案**：以管理员身份运行命令行

---

**成功标志**：当您看到以下输出时，说明修复成功：
```
Error: Some required dependencies are missing:
  - mne, scipy (for data loading)
  - torch (for neural network models)

Install with: pip install -r requirements.txt
```

这表明我们的安全导入机制正在工作！