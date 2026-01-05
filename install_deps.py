#!/usr/bin/env python
"""
安装依赖脚本
"""
import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"执行: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True, encoding='utf-8')
        print("✅ 成功!")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 失败!")
        print(f"错误: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr.strip()}")
        return False

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"当前版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    print("✅ Python版本符合要求")
    return True

def check_pip():
    """检查pip是否可用"""
    print("\n📦 检查pip...")
    try:
        import pip
        print("✅ pip可用")
        return True
    except ImportError:
        print("❌ pip不可用")
        return False

def install_core_packages():
    """安装核心包"""
    core_packages = [
        ("numpy", "数值计算"),
        ("matplotlib", "绘图"),
        ("scipy", "科学计算"),
        ("pathlib", "路径处理"),
    ]
    
    print("\n📚 安装核心包...")
    success_count = 0
    
    for package, desc in core_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"安装{package} ({desc})"):
            success_count += 1
    
    print(f"\n核心包安装完成: {success_count}/{len(core_packages)}")
    return success_count == len(core_packages)

def install_ml_packages():
    """安装机器学习包"""
    ml_packages = [
        ("torch", "PyTorch深度学习框架"),
        ("torchvision", "PyTorch视觉工具"),
        ("scikit-learn", "机器学习工具"),
        ("pandas", "数据处理"),
        ("seaborn", "统计可视化"),
        ("tqdm", "进度条"),
    ]
    
    print("\n🤖 安装机器学习包...")
    success_count = 0
    
    for package, desc in ml_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"安装{package} ({desc})"):
            success_count += 1
    
    print(f"\n机器学习包安装完成: {success_count}/{len(ml_packages)}")
    return success_count >= len(ml_packages) * 0.8  # 80%成功率即可

def install_neuro_packages():
    """安装神经科学专用包"""
    neuro_packages = [
        ("mne", "MEG/EEG数据处理"),
        ("pywt", "小波变换"),
        ("optuna", "超参数优化"),
        ("pyyaml", "YAML配置文件"),
        ("plotly", "交互式可视化"),
        ("h5py", "HDF5文件格式"),
        ("joblib", "并行处理"),
        ("pytest", "测试框架"),
        ("black", "代码格式化"),
        ("flake8", "代码检查"),
    ]
    
    print("\n🧠 安装神经科学专用包...")
    success_count = 0
    
    for package, desc in neuro_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"安装{package} ({desc})"):
            success_count += 1
    
    print(f"\n神经科学包安装完成: {success_count}/{len(neuro_packages)}")
    return success_count >= len(neuro_packages) * 0.7  # 70%成功率即可

def test_imports():
    """测试关键导入"""
    print("\n🧪 测试导入...")
    
    test_modules = [
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("scipy", None),
        ("pathlib", "Path"),
        ("logging", None),
        ("json", None),
        ("yaml", None),
        ("torch", "torch"),
        ("sklearn.model_selection", None),
        ("optuna", None),
        ("mne", None),
    ]
    
    success_count = 0
    for module_name, alias in test_modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✅ {module_name}")
            success_count += 1
        except ImportError:
            print(f"❌ {module_name}")
    
    print(f"\n导入测试: {success_count}/{len(test_modules)} 模块可用")
    return success_count >= len(test_modules) * 0.7

def create_simple_demo():
    """创建简化的演示"""
    print("\n🎯 创建演示文件...")
    
    demo_content = '''#!/usr/bin/env python
"""
BCI项目演示 - 简化版
"""

print("🧠 多模态脑机接口运动意图解码系统")
print("=" * 50)

print("\n📁 项目结构:")
import os
for root, dirs, files in os.walk("."):
    level = root.replace(".", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        if not file.startswith('.'):
            print(f"{subindent}{file}")
    if level == 2:  # 只显示两层
        break

print("\\n🔧 核心功能:")
print("• 数据预处理: EEG和fNIRS信号处理")
print("• 特征提取: 时频分析和血流动力学特征")
print("• 模型训练: 多模态深度学习")
print("• 实时推理: <200ms延迟")
print("• 性能评估: 全面评估指标")

print("\\n📊 预期性能:")
print("• 分类准确率: >85%")
print("• 信息传输率: >18 bits/min")
print("• 实时延迟: <200ms")

print("\\n🚀 下一步:")
print("1. 准备数据集到 data/raw/ 目录")
print("2. 运行: python main.py --mode experiment")
print("3. 查看结果: results/ 目录")

print("\\n" + "=" * 50)
print("项目已准备就绪! 🎉")
print("=" * 50)
'''
    
    with open('demo_simple.py', 'w', encoding='utf-8') as f:
        f.write(demo_content)
    
    print("✅ 演示文件已创建: demo_simple.py")

def main():
    """主安装函数"""
    print("=" * 60)
    print("🧠 多模态BCI项目依赖安装器")
    print("=" * 60)
    
    # 1. 检查Python版本
    if not check_python_version():
        return
    
    # 2. 检查pip
    if not check_pip():
        print("❌ 请先安装pip")
        return
    
    # 3. 升级pip
    run_command(f"{sys.executable} -m pip install --upgrade pip", "升级pip")
    
    # 4. 安装核心包
    if not install_core_packages():
        print("❌ 核心包安装失败")
        return
    
    # 5. 安装机器学习包
    ml_success = install_ml_packages()
    
    # 6. 安装神经科学包
    neuro_success = install_neuro_packages()
    
    # 7. 测试导入
    import_success = test_imports()
    
    # 8. 创建演示
    create_simple_demo()
    
    # 9. 总结
    print("\n" + "=" * 60)
    print("📋 安装总结")
    print("=" * 60)
    
    print(f"🔧 核心包: ✅")
    print(f"🤖 机器学习包: {'✅' if ml_success else '⚠️ 部分'}")
    print(f"🧠 神经科学包: {'✅' if neuro_success else '⚠️ 部分'}")
    print(f"🧪 导入测试: {'✅' if import_success else '⚠️ 部分'}")
    
    print("\n🎯 下一步操作:")
    print("1. 运行演示: python demo_simple.py")
    print("2. 准备数据集到 data/raw/ 目录")
    print("3. 运行实验: python main.py --mode experiment")
    print("4. 查看文档: README.md")
    
    if not (ml_success and neuro_success and import_success):
        print("\n⚠️  部分包安装失败，项目仍可运行基础功能")
        print("完整功能需要所有依赖包。")
    
    print("\n🎉 安装完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()