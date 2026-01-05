#!/usr/bin/env python
"""
快速修复脚本 - 解决pip问题并安装基础依赖
"""
import subprocess
import sys
import os

def run_command(cmd, description, timeout=60):
    """运行命令"""
    print(f"\n🔧 {description}")
    print(f"执行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("✅ 成功!")
            return True
        else:
            print("❌ 失败!")
            print(f"错误代码: {result.returncode}")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 BCI项目快速修复")
    print("=" * 60)
    
    # 检查当前Python环境
    print(f"\n🐍 Python路径: {sys.executable}")
    print(f"🐍 Python版本: {sys.version}")
    
    # 方法1: 尝试直接使用pip命令
    print("\n📦 方法1: 尝试直接pip命令...")
    if run_command("pip --version", "检查pip版本"):
        print("✅ pip可直接使用")
        # 尝试安装numpy作为测试
        if run_command("pip install numpy", "安装numpy测试"):
            print("✅ pip工作正常，可以正常安装包")
            return
        else:
            print("❌ pip工作异常")
    
    # 方法2: 尝试python -m pip
    print("\n📦 方法2: 尝试python -m pip...")
    if run_command(f"{sys.executable} -m pip --version", "检查python -m pip"):
        print("✅ python -m pip可用")
        if run_command(f"{sys.executable} -m pip install numpy", "安装numpy测试"):
            print("✅ python -m pip工作正常")
            return
    
    # 方法3: 检查是否有其他Python安装
    print("\n📦 方法3: 查找其他Python安装...")
    python_commands = ["python3", "py", "python"]
    
    for py_cmd in python_commands:
        result = subprocess.run(f"{py_cmd} --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 找到Python: {py_cmd}")
            print(f"版本: {result.stdout.strip()}")
            
            # 测试这个Python的pip
            if run_command(f"{py_cmd} -m pip --version", f"检查{py_cmd}的pip"):
                print(f"✅ {py_cmd}的pip可用")
                
                # 安装基础包
                packages = ["numpy", "matplotlib", "scipy"]
                for pkg in packages:
                    run_command(f"{py_cmd} -m pip install {pkg}", f"安装{pkg}")
                
                print(f"\n🎯 使用方法: {py_cmd} main.py --mode experiment")
                return
    
    # 方法4: 提供手动解决方案
    print("\n❌ 所有自动方法都失败了")
    print("\n📋 手动解决方案:")
    print("1. 打开命令提示符(CMD)或PowerShell(管理员)")
    print("2. 运行以下命令:")
    print("   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py")
    print("   python get-pip.py")
    print("3. 然后运行:")
    print("   pip install numpy matplotlib scipy torch scikit-learn mne")
    
    print("\n📦 或者使用conda:")
    print("1. 安装Miniconda或Anaconda")
    print("2. 创建环境: conda create -n bci python=3.12")
    print("3. 激活环境: conda activate bci")
    print("4. 安装依赖: conda install numpy matplotlib scipy scikit-learn pytorch")
    print("5. 安装mne: pip install mne")
    
    print("\n🔧 或者重新安装Python:")
    print("1. 从python.org下载Python安装包")
    print("2. 安装时勾选'Add to PATH'和'Install pip'")
    print("3. 重启后重新运行此脚本")

if __name__ == "__main__":
    main()