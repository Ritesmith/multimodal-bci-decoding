#!/usr/bin/env python
"""
基础测试脚本 - 不依赖外部库
"""
import os
import sys
import json
from pathlib import Path

def show_project_overview():
    """显示项目概览"""
    print("🧠 多模态脑机接口运动意图解码系统")
    print("=" * 60)
    
    print("\n📁 当前目录结构:")
    base_path = Path(".")
    
    # 显示文件和目录
    for item in sorted(base_path.iterdir()):
        if item.name.startswith('.'):
            continue
        
        if item.is_file():
            size = item.stat().st_size
            print(f"  📄 {item.name:<30} ({size:>5} bytes)")
        elif item.is_dir():
            try:
                count = len(list(item.iterdir()))
                print(f"  📁 {item.name:<30} ({count:>3} items)")
            except:
                print(f"  📁 {item.name:<30} (access denied)")

def check_core_files():
    """检查核心文件"""
    print("\n🔍 核心文件检查:")
    
    required_files = [
        "config.py",
        "main.py", 
        "README.md",
        "requirements.txt",
        "experiment_config.yaml",
        ".gitignore"
    ]
    
    all_exist = True
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name}")
            all_exist = False
    
    return all_exist

def check_src_structure():
    """检查src目录结构"""
    print("\n📦 src模块检查:")
    
    src_path = Path("src")
    if not src_path.exists():
        print("  ❌ src目录不存在")
        return False
    
    required_modules = [
        "__init__.py",
        "data_loader.py",
        "feature_extraction.py", 
        "models.py",
        "training.py",
        "evaluation.py",
        "realtime.py",
        "experiment.py"
    ]
    
    all_exist = True
    for module_name in required_modules:
        module_path = src_path / module_name
        if module_path.exists():
            size = module_path.stat().st_size
            print(f"  ✅ {module_name:<25} ({size:>6} bytes)")
        else:
            print(f"  ❌ {module_name:<25}")
            all_exist = False
    
    return all_exist

def show_project_features():
    """显示项目特性"""
    print("\n🌟 项目特性:")
    
    features = [
        "🧠 多模态融合: EEG + fNIRS神经成像",
        "🔄 实时解码: 支持200ms以内推理延迟", 
        "📊 全面评估: 准确率、ITR、F1分数",
        "🔧 灵活配置: 支持多种网络架构",
        "📈 可视化分析: 丰富的分析工具",
        "⚡ 高性能: CNN-LSTM双流架构",
        "🎯 运动意图解码: 4类运动想象任务",
        "🔍 注意力机制: 动态加权融合策略"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")

def show_technical_architecture():
    """显示技术架构"""
    print("\n🏗️ 技术架构:")
    
    architecture = {
        "数据处理": [
            "EEG预处理: 滤波、ICA、伪迹去除",
            "fNIRS预处理: 运动校正、基线校正", 
            "同步对齐: 时间戳对齐、延迟补偿"
        ],
        "特征提取": [
            "EEG特征: 时频分析、PLV、ERD/ERS",
            "fNIRS特征: 血流动力学响应、空间特征",
            "融合策略: 早期、晚期、注意力融合"
        ],
        "模型架构": [
            "EEG分支: 3D卷积 + LSTM时序建模",
            "fNIRS分支: 全连接网络 + 压缩降维",
            "融合层: 注意力机制 + Softmax分类"
        ],
        "评估指标": [
            "分类性能: 准确率、精确率、召回率、F1",
            "实时性能: 推理延迟、信息传输率",
            "可视化: 混淆矩阵、激活图、决策边界"
        ]
    }
    
    for category, items in architecture.items():
        print(f"\n  {category}:")
        for item in items:
            print(f"    • {item}")

def create_status_report():
    """创建状态报告"""
    print("\n📋 生成状态报告...")
    
    core_files_ok = check_core_files()
    src_structure_ok = check_src_structure()
    
    status = {
        "project_name": "多模态脑机接口运动意图解码系统",
        "status": "ready" if core_files_ok and src_structure_ok else "incomplete",
        "core_files": core_files_ok,
        "src_structure": src_structure_ok,
        "modules_count": 8,
        "features": [
            "多模态融合",
            "实时解码",
            "全面评估", 
            "可视化分析",
            "参数调优"
        ],
        "performance_targets": {
            "accuracy": ">85%",
            "itr": ">18 bits/min",
            "latency": "<200ms"
        },
        "supported_datasets": [
            "BNCI Horizon 2020",
            "OpenBMI"
        ]
    }
    
    # 保存状态报告
    with open("project_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    print("✅ 状态报告已保存到 project_status.json")
    return status

def show_usage_instructions():
    """显示使用说明"""
    print("\n🚀 使用说明:")
    
    instructions = [
        ("环境准备", "安装Python 3.8+ 和必要依赖包"),
        ("数据准备", "下载BNCI或OpenBMI数据集到 data/raw/"),
        ("运行实验", "python main.py --mode experiment"),
        ("训练模型", "python main.py --mode train"),
        ("评估性能", "python main.py --mode eval"),
        ("实时测试", "python main.py --mode realtime")
    ]
    
    for step, desc in instructions:
        print(f"  {step:<12} - {desc}")
    
    print("\n📚 文档资源:")
    print("  • README.md - 项目概述和安装指南")
    print("  • USAGE_GUIDE.md - 详细使用说明")
    print("  • experiment_config.yaml - 实验配置文件")
    print("  • requirements.txt - 依赖包列表")

def main():
    """主函数"""
    print("=" * 60)
    print("🧠 BCI项目基础检查器")
    print("=" * 60)
    
    # 1. 显示项目概览
    show_project_overview()
    
    # 2. 检查核心文件
    core_files_ok = check_core_files()
    
    # 3. 检查src结构
    src_structure_ok = check_src_structure()
    
    # 4. 显示项目特性
    show_project_features()
    
    # 5. 显示技术架构
    show_technical_architecture()
    
    # 6. 创建状态报告
    status = create_status_report()
    
    # 7. 显示使用说明
    show_usage_instructions()
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)
    
    print(f"🔧 核心文件: {'✅ 完整' if core_files_ok else '❌ 缺失'}")
    print(f"📦 源码结构: {'✅ 完整' if src_structure_ok else '❌ 缺失'}")
    print(f"📋 项目状态: {status['status'].upper()}")
    
    if core_files_ok and src_structure_ok:
        print("\n🎉 项目结构完整，可以开始使用!")
        print("📝 下一步:")
        print("  1. 安装依赖: pip install -r requirements.txt")
        print("  2. 准备数据集到 data/raw/")
        print("  3. 运行实验: python main.py --mode experiment")
    else:
        print("\n⚠️  项目结构不完整，请检查缺失的文件")
    
    print("=" * 60)

if __name__ == "__main__":
    main()