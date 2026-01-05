#!/usr/bin/env python
"""
直接依赖安装脚本 - 不依赖pip模块
"""
import subprocess
import sys
import os

def run_pip_command(package, description):
    """直接使用pip命令安装包"""
    print(f"\n🔧 {description}")
    print(f"安装: {package}")
    
    try:
        # 尝试不同的pip命令
        commands = [
            f"pip install {package}",
            f"pip3 install {package}",
            f"python -m pip install {package}",
            f"python3 -m pip install {package}",
            f"{sys.executable} -m pip install {package}"
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"✅ 成功安装: {package}")
                    if result.stdout:
                        print(f"输出: {result.stdout[:200]}...")
                    return True
                else:
                    print(f"❌ 命令失败: {cmd}")
                    if result.stderr:
                        print(f"错误: {result.stderr[:200]}...")
            except subprocess.TimeoutExpired:
                print(f"❌ 安装超时: {package}")
                continue
            except Exception as e:
                print(f"❌ 异常: {e}")
                continue
        
        print(f"❌ 无法安装: {package}")
        return False
        
    except Exception as e:
        print(f"❌ 安装 {package} 时发生错误: {e}")
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

def check_pip_available():
    """检查pip是否可用"""
    print("\n📦 检查pip可用性...")
    
    pip_commands = ["pip", "pip3", f"{sys.executable} -m pip"]
    
    for cmd in pip_commands:
        try:
            result = subprocess.run(f"{cmd} --version", shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ pip可用: {cmd}")
                print(f"版本: {result.stdout.strip()}")
                return True
        except:
            continue
    
    print("❌ pip不可用")
    print("\n请手动安装pip:")
    print("1. 下载 get-pip.py: https://bootstrap.pypa.io/get-pip.py")
    print("2. 运行: python get-pip.py")
    return False

def install_essential_packages():
    """安装最基础的包"""
    essential = [
        ("numpy", "数值计算基础库"),
        ("setuptools", "Python包管理工具"),
        ("wheel", "Python包构建工具"),
    ]
    
    print("\n📚 安装基础工具包...")
    success = 0
    
    for package, desc in essential:
        if run_pip_command(package, f"安装{package} ({desc})"):
            success += 1
    
    print(f"\n基础包安装: {success}/{len(essential)}")
    return success >= 2  # 至少安装2个

def install_core_dependencies():
    """安装核心依赖"""
    core_deps = [
        ("matplotlib", "绘图库"),
        ("scipy", "科学计算"),
        ("scikit-learn", "机器学习"),
        ("pandas", "数据处理"),
        ("torch", "PyTorch深度学习"),
        ("tqdm", "进度条"),
    ]
    
    print("\n🔧 安装核心依赖...")
    success = 0
    
    for package, desc in core_deps:
        if run_pip_command(package, f"安装{package} ({desc})"):
            success += 1
    
    print(f"\n核心依赖安装: {success}/{len(core_deps)}")
    return success >= len(core_deps) * 0.6  # 60%成功率

def install_neuro_dependencies():
    """安装神经科学相关依赖"""
    neuro_deps = [
        ("mne", "EEG/MEG数据处理"),
        ("pywt", "小波变换"),
        ("seaborn", "统计可视化"),
        ("pyyaml", "YAML配置"),
        ("h5py", "HDF5文件支持"),
        ("plotly", "交互式图表"),
        ("optuna", "超参数优化"),
    ]
    
    print("\n🧠 安装神经科学依赖...")
    success = 0
    
    for package, desc in neuro_deps:
        if run_pip_command(package, f"安装{package} ({desc})"):
            success += 1
    
    print(f"\n神经科学依赖安装: {success}/{len(neuro_deps)}")
    return success >= len(neuro_deps) * 0.5  # 50%成功率

def test_basic_imports():
    """测试基础导入"""
    print("\n🧪 测试基础导入...")
    
    basic_tests = [
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("scipy", None),
        ("pathlib", "Path"),
        ("logging", None),
        ("json", None),
    ]
    
    success = 0
    for module, alias in basic_tests:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
            success += 1
        except ImportError:
            print(f"❌ {module}")
    
    print(f"\n基础导入测试: {success}/{len(basic_tests)}")
    return success >= len(basic_tests) * 0.8

def main():
    """主安装流程"""
    print("=" * 60)
    print("🧠 多模态BCI项目 - 直接依赖安装器")
    print("=" * 60)
    
    # 1. 检查Python版本
    if not check_python_version():
        return
    
    # 2. 检查pip可用性
    if not check_pip_available():
        print("\n❌ pip不可用，无法继续安装")
        print("\n解决方案:")
        print("1. 确保使用Python安装包中的pip")
        print("2. 或使用conda环境: conda install pip")
        print("3. 重新安装Python时确保包含pip")
        return
    
    # 3. 安装基础工具
    if not install_essential_packages():
        print("❌ 基础工具包安装失败")
        return
    
    # 4. 安装核心依赖
    core_success = install_core_dependencies()
    
    # 5. 安装神经科学依赖
    neuro_success = install_neuro_dependencies()
    
    # 6. 测试基础导入
    import_success = test_basic_imports()
    
    # 7. 总结
    print("\n" + "=" * 60)
    print("📋 安装总结")
    print("=" * 60)
    
    print(f"🔧 基础工具: ✅")
    print(f"💻 核心依赖: {'✅' if core_success else '⚠️ 部分'}")
    print(f"🧠 神经科学: {'✅' if neuro_success else '⚠️ 部分'}")
    print(f"🧪 导入测试: {'✅' if import_success else '❌'}")
    
    print("\n🎯 下一步:")
    if import_success:
        print("✅ 基础依赖安装成功!")
        print("1. 运行: python main.py --mode experiment")
        print("2. 如果仍有错误，可能需要安装额外依赖")
    else:
        print("❌ 基础导入测试失败")
        print("请检查Python环境和pip配置")
    
    print("\n💡 提示:")
    print("- 如果某些包安装失败，可以单独安装")
    print("- 使用管理员权限运行可能有助于安装")
    print("- 考虑使用虚拟环境避免冲突")
    
    print("\n🎉 安装过程完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()