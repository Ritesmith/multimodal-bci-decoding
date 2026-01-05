#!/usr/bin/env python
"""
手动安装pip脚本 - 不依赖任何包管理器
"""
import urllib.request
import sys
import os
import subprocess
import tempfile

def download_file(url, filename):
    """下载文件"""
    print(f"📥 下载: {url}")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ 下载完成: {filename}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def run_python_script(script_path, description):
    """运行Python脚本"""
    print(f"\n🐍 执行: {description}")
    cmd = f"{sys.executable} {script_path}"
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("✅ 执行完成!")
        if result.stdout:
            print(f"输出: {result.stdout}")
        if result.stderr and "WARNING" in result.stderr:
            print(f"警告: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

def install_pip_manually():
    """手动安装pip"""
    print("=" * 60)
    print("🔧 手动安装pip")
    print("=" * 60)
    
    # 方法1: 使用urllib直接下载get-pip.py
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    temp_dir = tempfile.gettempdir()
    get_pip_path = os.path.join(temp_dir, "get-pip.py")
    
    print(f"\n📥 下载pip安装器到: {get_pip_path}")
    
    if download_file(get_pip_url, get_pip_path):
        print(f"\n🐍 运行pip安装器...")
        if run_python_script(get_pip_path, "安装pip"):
            print("✅ pip安装成功!")
            return True
        else:
            print("❌ pip安装失败!")
    
    return False

def test_pip_installation():
    """测试pip是否安装成功"""
    print("\n🧪 测试pip安装...")
    
    # 等待一下让系统识别新安装的pip
    import time
    time.sleep(2)
    
    test_commands = [
        f"{sys.executable} -m pip --version",
        "pip --version",
        "pip3 --version"
    ]
    
    for cmd in test_commands:
        print(f"\n🔍 测试: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ pip可用!")
                print(f"版本: {result.stdout.strip()}")
                return cmd
        except:
            continue
    
    print("❌ pip仍不可用")
    return None

def install_basic_packages(pip_command):
    """安装基础包"""
    print(f"\n📦 使用 {pip_command} 安装基础包...")
    
    packages = [
        ("numpy", "数值计算"),
        ("matplotlib", "绘图"),
        ("scipy", "科学计算"),
        ("setuptools", "包管理"),
        ("wheel", "构建工具")
    ]
    
    success_count = 0
    for package, desc in packages:
        print(f"\n🔧 安装 {package} ({desc})...")
        cmd = f"{pip_command} install {package}"
        print(f"命令: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"✅ {package} 安装成功!")
                success_count += 1
            else:
                print(f"❌ {package} 安装失败!")
                if result.stderr:
                    print(f"错误: {result.stderr[:300]}")
        except Exception as e:
            print(f"❌ {package} 安装异常: {e}")
    
    print(f"\n📊 安装统计: {success_count}/{len(packages)} 成功")
    return success_count >= len(packages) * 0.6

def test_basic_imports():
    """测试基础导入"""
    print("\n🧪 测试基础包导入...")
    
    basic_tests = [
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("scipy", None),
        ("pathlib", "Path"),
        ("json", None),
        ("logging", None)
    ]
    
    success_count = 0
    for module, alias in basic_tests:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print(f"\n📊 导入测试: {success_count}/{len(basic_tests)} 成功")
    return success_count >= len(basic_tests) * 0.7

def main():
    """主安装流程"""
    print("🚀 手动pip安装器")
    print("=" * 60)
    
    # 1. 显示环境信息
    print(f"🐍 Python路径: {sys.executable}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📁 临时目录: {tempfile.gettempdir()}")
    
    # 2. 手动安装pip
    if not install_pip_manually():
        print("❌ pip安装失败")
        print("\n📋 备选方案:")
        print("1. 从浏览器下载: https://bootstrap.pypa.io/get-pip.py")
        print("2. 保存为 get-pip.py")
        print("3. 手动运行: python get-pip.py")
        return
    
    # 3. 测试pip安装
    pip_cmd = test_pip_installation()
    if not pip_cmd:
        print("❌ pip安装验证失败")
        return
    
    # 4. 安装基础包
    if not install_basic_packages(pip_cmd):
        print("⚠️ 基础包安装部分失败，但pip已可用")
    
    # 5. 测试导入
    import_success = test_basic_imports()
    
    # 6. 总结
    print("\n" + "=" * 60)
    print("📋 安装总结")
    print("=" * 60)
    
    if pip_cmd:
        print("✅ pip: 安装成功")
        print(f"📦 命令: {pip_cmd}")
    else:
        print("❌ pip: 安装失败")
    
    if import_success:
        print("✅ 基础包: 可用")
        print("\n🎯 下一步:")
        print("1. 测试修复效果: python test_fix.py")
        print("2. 运行项目: python main.py --mode experiment")
        print("3. 安装更多包: pip install torch scikit-learn mne")
    else:
        print("⚠️ 基础包: 部分可用")
        print("\n🔧 需要手动安装:")
        print(f"{pip_cmd} install numpy matplotlib scipy")
    
    print("\n💡 提示:")
    print("- 如果pip命令仍不可用，重启命令行再试")
    print("- 使用管理员权限可能有助于安装")
    print("- 可以逐个安装包来排查问题")
    
    print("\n🎉 手动安装完成!")

if __name__ == "__main__":
    main()