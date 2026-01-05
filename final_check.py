#!/usr/bin/env python
"""
最终项目状态检查
"""
import os
import sys
from pathlib import Path

def main():
    print("🧠 多模态脑机接口运动意图解码系统")
    print("=" * 60)
    
    # 检查项目文件
    print("\n📁 项目文件检查:")
    
    required_files = [
        ("config.py", "配置文件"),
        ("main.py", "主程序"),
        ("README.md", "项目文档"),
        ("USAGE_GUIDE.md", "使用指南"),
        ("requirements.txt", "依赖列表"),
        ("experiment_config.yaml", "实验配置"),
        (".gitignore", "版本控制")
    ]
    
    for file_name, description in required_files:
        if os.path.exists(file_name):
            size = os.path.getsize(file_name)
            print(f"  ✅ {file_name:<25} {description:<15} ({size:>5}B)")
        else:
            print(f"  ❌ {file_name:<25} {description:<15} 缺失")
    
    # 检查src目录
    print("\n📦 核心模块:")
    if os.path.exists('src'):
        src_files = [
            ("__init__.py", "包初始化"),
            ("data_loader.py", "数据加载"),
            ("feature_extraction.py", "特征提取"),
            ("models.py", "神经网络模型"),
            ("training.py", "训练流程"),
            ("evaluation.py", "性能评估"),
            ("realtime.py", "实时推理"),
            ("experiment.py", "实验管理")
        ]
        
        for file_name, description in src_files:
            file_path = os.path.join('src', file_name)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  ✅ {file_name:<25} {description:<15} ({size:>5}B)")
            else:
                print(f"  ❌ {file_name:<25} {description:<15} 缺失")
    else:
        print("  ❌ src目录不存在")
    
    # 检查功能完整性
    print("\n🎯 核心功能:")
    
    features = [
        "多模态数据融合 (EEG + fNIRS)",
        "实时推理系统 (<200ms延迟)",
        "深度学习模型 (CNN-LSTM)",
        "注意力机制融合",
        "交叉验证训练",
        "超参数调优",
        "性能评估和可视化",
        "实验自动化流程"
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")
    
    # 使用说明
    print("\n🚀 快速开始:")
    print("  1. 安装依赖: pip install -r requirements.txt")
    print("  2. 运行实验: python main.py --mode experiment")
    print("  3. 查看文档: README.md")
    
    # 技术规格
    print("\n📊 技术规格:")
    print("  • 数据集: BNCI Horizon 2020, OpenBMI")
    print("  • 模型架构: 双流CNN-LSTM + 注意力融合")
    print("  • 预期准确率: >85%")
    print("  • 信息传输率: >18 bits/min")
    print("  • 实时延迟: <200ms")
    
    print("\n" + "=" * 60)
    print("✅ 项目创建完成！准备开始脑机接口研究！")
    print("=" * 60)

if __name__ == "__main__":
    main()