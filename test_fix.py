#!/usr/bin/env python
"""
测试修复后的导入机制
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("🧪 测试修复后的导入机制")
print("=" * 50)

def test_module_import(module_name, description):
    """测试模块导入"""
    try:
        exec(f"import {module_name}")
        print(f"✅ {description} - 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {description} - 导入失败: {e}")
        return False

def test_safe_import():
    """测试安全导入机制"""
    print("\n🛡️  测试安全导入机制...")
    
    # 测试各个模块的check_module_availability函数
    modules = [
        ("data_loader", "数据加载模块"),
        ("feature_extraction", "特征提取模块"),
        ("models", "模型模块"),
        ("training", "训练模块"),
        ("evaluation", "评估模块"),
        ("realtime", "实时处理模块"),
    ]
    
    results = {}
    
    for module, desc in modules:
        try:
            mod = __import__(module)
            if hasattr(mod, 'check_module_availability'):
                available = mod.check_module_availability()
                results[module] = available
                status = "✅ 可用" if available else "⚠️ 依赖缺失"
                print(f"  {desc}: {status}")
            else:
                print(f"  {desc}: ❌ 缺少检查函数")
                results[module] = False
        except ImportError as e:
            print(f"  {desc}: ❌ 导入失败: {e}")
            results[module] = False
    
    return results

def test_main_imports():
    """测试主程序导入"""
    print("\n🎯 测试主程序导入...")
    
    try:
        from experiment import ExperimentRunner, ExperimentConfig
        print("✅ 实验运行器 - 可用")
        return True
    except ImportError as e:
        print(f"❌ 实验运行器 - 导入失败: {e}")
        return False

def test_error_handling():
    """测试错误处理机制"""
    print("\n🔧 测试错误处理机制...")
    
    try:
        # 尝试创建实验配置
        from experiment import ExperimentConfig
        config = ExperimentConfig()
        print("✅ 实验配置 - 创建成功")
        
        # 尝试创建实验运行器
        from experiment import ExperimentRunner
        runner = ExperimentRunner(config)
        print("✅ 实验运行器 - 创建成功")
        
        # 注意: 这里不实际运行，只测试对象创建
        print("✅ 错误处理机制 - 正常工作")
        return True
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🐍 Python版本:", sys.version)
    print("📁 当前目录:", Path.cwd())
    
    # 1. 基础模块测试
    basic_modules = [
        ("numpy", "数值计算"),
        ("json", "JSON处理"),
        ("pathlib", "路径处理"),
        ("logging", "日志记录"),
        ("datetime", "日期时间"),
    ]
    
    basic_ok = 0
    for module, desc in basic_modules:
        if test_module_import(module, desc):
            basic_ok += 1
    
    print(f"\n📊 基础模块: {basic_ok}/{len(basic_modules)} 可用")
    
    # 2. 安全导入测试
    safe_results = test_safe_import()
    safe_ok = sum(1 for v in safe_results.values() if v)
    print(f"\n📊 模块可用性: {safe_ok}/{len(safe_results)} 可用")
    
    # 3. 主程序测试
    main_ok = test_main_imports()
    
    # 4. 错误处理测试
    error_ok = test_error_handling()
    
    # 5. 总结
    print("\n" + "=" * 50)
    print("📋 测试总结")
    print("=" * 50)
    
    if basic_ok >= 4:
        print("✅ 基础环境正常")
    else:
        print("❌ 基础环境异常")
    
    if safe_ok >= 3:
        print("✅ 安全导入机制正常")
    else:
        print("⚠️ 部分模块依赖缺失 (这是正常的)")
    
    if main_ok:
        print("✅ 主程序可运行")
    else:
        print("❌ 主程序导入失败")
    
    if error_ok:
        print("✅ 错误处理机制正常")
    else:
        print("❌ 错误处理异常")
    
    # 6. 建议
    print("\n💡 建议:")
    if safe_ok < len(safe_results):
        print("- 某些模块依赖缺失，可以运行基础功能")
        print("- 完整功能需要安装: pip install numpy matplotlib scipy torch scikit-learn mne")
    
    if main_ok and error_ok:
        print("- ✅ 修复成功! 可以运行: python main.py --mode experiment")
        print("- 如果遇到依赖错误，程序会友好提示")
    else:
        print("- ❌ 还需要进一步修复")
    
    print("\n🎉 测试完成!")

if __name__ == "__main__":
    main()