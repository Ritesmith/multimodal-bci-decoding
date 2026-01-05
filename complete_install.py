#!/usr/bin/env python
"""
完整安装脚本 - 完成依赖安装并测试
"""
import subprocess
import sys

def run_command(cmd, description, timeout=120):
    """运行命令"""
    print(f"\n🔧 {description}")
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            print("✅ 成功!")
            return True
        else:
            print("❌ 失败!")
            if result.stderr:
                print(f"错误: {result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 超时!")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def install_core_packages():
    """安装核心包"""
    core_packages = [
        "torch",           # PyTorch
        "scikit-learn",    # 机器学习
        "plotly",          # 交互式图表
        "h5py",           # HDF5文件
        "joblib",          # 并行处理
        "pywt",            # 小波变换
        "tqdm",            # 进度条
        "seaborn",         # 统计可视化
    ]
    
    print("📦 安装核心依赖包...")
    success = 0
    
    for package in core_packages:
        if run_command(f"pip install {package}", f"安装{package}"):
            success += 1
    
    print(f"\n📊 安装统计: {success}/{len(core_packages)} 成功")
    return success >= len(core_packages) * 0.8

def test_imports():
    """测试导入"""
    print("\n🧪 测试模块导入...")
    
    test_modules = [
        ("numpy", "✅ 基础数值计算"),
        ("matplotlib.pyplot", "✅ 绘图库"),
        ("scipy", "✅ 科学计算"),
        ("torch", "🔥 PyTorch深度学习"),
        ("sklearn", "🤖 机器学习工具"),
        ("mne", "🧠 EEG数据处理"),
        ("plotly", "📊 交互式图表"),
        ("h5py", "📁 HDF5文件支持"),
        ("pywt", "🌊 小波变换"),
        ("tqdm", "📊 进度条"),
        ("seaborn", "📈 统计可视化"),
    ]
    
    success = 0
    for module, desc in test_modules:
        try:
            exec(f"import {module}")
            print(f"{desc}")
            success += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print(f"\n📊 导入测试: {success}/{len(test_modules)} 成功")
    return success >= len(test_modules) * 0.7

def test_main_program():
    """测试主程序"""
    print("\n🎯 测试主程序...")
    
    try:
        # 测试实验模块导入
        sys.path.append('src')
        from experiment import ExperimentRunner, ExperimentConfig
        print("✅ 实验运行器 - 导入成功")
        
        # 测试创建配置
        config = ExperimentConfig()
        print("✅ 实验配置 - 创建成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def main():
    """主流程"""
    print("=" * 60)
    print("🎯 BCI项目依赖安装完成器")
    print("=" * 60)
    
    print(f"🐍 Python: {sys.executable}")
    
    # 1. 安装核心包
    if not install_core_packages():
        print("⚠️ 部分包安装失败，但继续测试...")
    
    # 2. 测试导入
    import_success = test_imports()
    
    # 3. 测试主程序
    main_success = test_main_program()
    
    # 4. 总结
    print("\n" + "=" * 60)
    print("📋 最终总结")
    print("=" * 60)
    
    if import_success:
        print("✅ 依赖导入: 成功")
        print("🎯 可以运行: python main.py --mode experiment")
        if main_success:
            print("🎉 主程序: 完全可用")
        else:
            print("⚠️ 主程序: 部分可用")
    else:
        print("❌ 依赖导入: 失败")
        print("🔧 请检查安装的包")
    
    print("\n💡 下一步:")
    if import_success and main_success:
        print("✅ 运行实验: python main.py --mode experiment")
        print("📊 查看结果: results/ 目录")
    else:
        print("🔧 手动安装缺失包")
        print("📋 查看错误日志")
    
    print("\n🎉 安装完成!")

if __name__ == "__main__":
    main()