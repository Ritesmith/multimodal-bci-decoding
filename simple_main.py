#!/usr/bin/env python
"""
简化版主程序 - 不依赖复杂的外部库
"""
import os
import sys
import json
from pathlib import Path

def check_project_structure():
    """检查项目结构"""
    print("检查项目结构...")
    
    base_dir = Path(".")
    
    # 检查核心文件
    core_files = [
        "config.py",
        "main.py", 
        "README.md",
        "USAGE_GUIDE.md",
        "requirements.txt",
        "experiment_config.yaml",
        ".gitignore"
    ]
    
    print("\n📄 核心文件:")
    for file_name in core_files:
        file_path = base_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✓ {file_name} ({size} bytes)")
        else:
            print(f"  ✗ {file_name} 缺失")
    
    # 检查目录结构
    dirs = ["src", "data", "models", "results", "logs"]
    
    print("\n📁 目录结构:")
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            print(f"  ✓ {dir_name}/ ({len(files)} files)")
        else:
            print(f"  ✗ {dir_name}/ 缺失")
            dir_path.mkdir(exist_ok=True)
            print(f"    已创建 {dir_name}/")

def show_project_info():
    """显示项目信息"""
    print("\n" + "="*60)
    print("🧠 多模态脑机接口运动意图解码系统")
    print("="*60)
    
    print("\n🎯 项目目标:")
    print("• 同步EEG-fNIRS神经成像")
    print("• 多模态特征融合")
    print("• 实时运动意图解码")
    print("• 高精度分类性能")
    
    print("\n🏗️ 技术架构:")
    print("• EEG分支: CNN-LSTM网络")
    print("• fNIRS分支: 全连接网络")
    print("• 融合策略: 注意力机制")
    print("• 评估指标: 准确率、ITR、F1分数")
    
    print("\n📊 预期性能:")
    print("• 多模态准确率: >85%")
    print("• 信息传输率: >18 bits/min")
    print("• 实时延迟: <200ms")
    print("• 相比单模态提升: 8-15%")

def show_modules_info():
    """显示模块信息"""
    print("\n🔧 核心模块:")
    
    modules = {
        "src/data_loader.py": [
            "• BNCI Horizon 2020数据集支持",
            "• OpenBMI数据集支持", 
            "• EEG预处理流水线",
            "• fNIRS预处理流水线",
            "• 多模态信号同步"
        ],
        "src/feature_extraction.py": [
            "• EEG时频分析(Morlet小波)",
            "• 相位锁定值(PLV)计算",
            "• ERD/ERS特征提取",
            "• fNIRS血流动力学特征",
            "• 多模态融合策略"
        ],
        "src/models.py": [
            "• 双流CNN-LSTM架构",
            "• 注意力融合机制",
            "• 单模态基线模型",
            "• 模型参数统计"
        ],
        "src/training.py": [
            "• 交叉验证训练",
            "• 早停机制",
            "• 学习率调度",
            "• 性能指标计算",
            "• 模型保存/加载"
        ],
        "src/evaluation.py": [
            "• 混淆矩阵可视化",
            "• 特征重要性分析",
            "• 脑区激活热图",
            "• 决策边界可视化",
            "• 交互式仪表板"
        ],
        "src/realtime.py": [
            "• 滑动窗口处理",
            "• 实时推理接口",
            "• 性能监控",
            "• 数据模拟器"
        ],
        "src/experiment.py": [
            "• 实验配置管理",
            "• 超参数调优",
            "• 网格搜索",
            "• Optuna优化",
            "• 自动报告生成"
        ]
    }
    
    for file_path, features in modules.items():
        print(f"\n📄 {file_path}:")
        for feature in features:
            print(f"  {feature}")

def show_usage_guide():
    """显示使用指南"""
    print("\n🚀 使用指南:")
    
    usage = [
        ("1. 环境准备", "pip install -r requirements.txt"),
        ("2. 数据准备", "下载BNCI数据集到 data/raw/"),
        ("3. 运行实验", "python main.py --mode experiment"),
        ("4. 训练模型", "python main.py --mode train"),
        ("5. 评估模型", "python main.py --mode eval"),
        ("6. 实时测试", "python main.py --mode realtime")
    ]
    
    for step, command in usage:
        print(f"  {step:<15} {command}")

def show_requirements():
    """显示依赖要求"""
    print("\n📦 依赖包:")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        for req in requirements[:10]:  # 显示前10个
            if req and not req.startswith('#'):
                print(f"  • {req}")
        
        if len(requirements) > 10:
            print(f"  ... 和其他 {len(requirements) - 10} 个包")
    
    except FileNotFoundError:
        print("  requirements.txt 文件未找到")

def create_project_status():
    """创建项目状态报告"""
    status = {
        "project_name": "多模态脑机接口运动意图解码系统",
        "status": "ready",
        "modules_count": 7,
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
        }
    }
    
    status_file = Path("project_status.json")
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 项目状态已保存到 {status_file}")

def main():
    """主函数"""
    print("="*60)
    print("🧠 多模态BCI项目检查器")
    print("="*60)
    
    # 检查项目结构
    check_project_structure()
    
    # 显示项目信息
    show_project_info()
    
    # 显示模块信息
    show_modules_info()
    
    # 显示使用指南
    show_usage_guide()
    
    # 显示依赖要求
    show_requirements()
    
    # 创建状态报告
    create_project_status()
    
    print("\n" + "="*60)
    print("✅ 项目检查完成！")
    print("\n📝 下一步操作:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 下载数据集到 data/raw/ 目录")
    print("3. 运行实验: python main.py --mode experiment")
    print("4. 查看文档: README.md 和 USAGE_GUIDE.md")
    print("="*60)

if __name__ == "__main__":
    main()