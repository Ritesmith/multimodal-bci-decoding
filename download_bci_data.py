#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BCI数据下载脚本
自动下载并组织真实BCI数据集到 data/raw/ 目录
"""

import os
import requests
import numpy as np
from scipy.io import savemat
import mne

def download_physionet_data():
    """下载PhysioNet EEG Motor Movement数据集"""
    print("🔍 正在下载PhysioNet EEG数据集...")
    
    try:
        # 下载subjects 1-10的运动想象数据
        subjects = list(range(1, 6))  # 先下载5个受试者
        runs = [4, 8, 12]  # 运动想象任务运行
        
        base_dir = "data/raw/physionet"
        os.makedirs(base_dir, exist_ok=True)
        
        for subject in subjects:
            print(f"  下载受试者 {subject:02d}...")
            try:
                # 使用MNE下载数据
                mne.datasets.eegbci.load_data(subjects=[subject], runs=runs, 
                                           path=base_dir, update_path=True)
                print(f"  ✅ 受试者 {subject:02d} 下载完成")
            except Exception as e:
                print(f"  ❌ 受试者 {subject:02d} 下载失败: {e}")
                continue
                
        return True
    except Exception as e:
        print(f"PhysioNet下载失败: {e}")
        return False

def create_realistic_sample_data():
    """创建更真实的样本数据供测试"""
    print("🧠 创建增强版样本数据...")
    
    base_dir = "data/raw/sample_enhanced"
    os.makedirs(base_dir, exist_ok=True)
    
    # 更真实的EEG参数
    fs_eeg = 1000  # Hz
    duration = 7   # seconds (-2s to +5s)
    n_channels = 64
    n_trials = 80  # 每类20个trial
    
    # 创建4类运动想象数据
    classes = ['left_hand', 'right_hand', 'feet', 'tongue']
    
    for class_idx, class_name in enumerate(classes):
        trials_data = []
        trials_labels = []
        
        for trial in range(20):  # 每类20个trial
            # 生成更真实的EEG信号
            t = np.linspace(0, duration, int(fs_eeg * duration))
            eeg_data = np.zeros((n_channels, len(t)))
            
            # 为每个通道添加不同频段的alpha/beta节律
            for ch in range(n_channels):
                # 基础alpha节律 (8-13Hz)
                alpha_freq = 8 + np.random.rand() * 5
                alpha_amp = 0.5 + np.random.rand() * 2
                eeg_data[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
                
                # beta节律 (13-30Hz) - 运动相关
                beta_freq = 13 + np.random.rand() * 17
                beta_amp = 0.3 + np.random.rand() * 1.5
                eeg_data[ch] += beta_amp * np.sin(2 * np.pi * beta_freq * t)
                
                # 类别特定特征 (ERD/ERS模式)
                if class_idx < 2:  # 手部运动
                    if ch < 32:  # 左半球
                        freq_mod = 20 if class_idx == 0 else 18
                        eeg_data[ch] += 1.5 * np.sin(2 * np.pi * freq_mod * t[1000:4000])
                    else:  # 右半球
                        freq_mod = 18 if class_idx == 0 else 20
                        eeg_data[ch] += 1.5 * np.sin(2 * np.pi * freq_mod * t[1000:4000])
                elif class_idx == 2:  # 脚部运动
                    # 中央区激活
                    if 20 <= ch <= 30:
                        eeg_data[ch] += 2.0 * np.sin(2 * np.pi * 25 * t[1000:4000])
                else:  # 舌部运动
                    # 额叶激活
                    if ch < 10:
                        eeg_data[ch] += 1.8 * np.sin(2 * np.pi * 22 * t[1000:4000])
                
                # 添加噪声
                eeg_data[ch] += 0.1 * np.random.randn(len(t))
            
            trials_data.append(eeg_data)
            trials_labels.append(class_idx)
        
        # 保存每个类别的数据
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        savemat(os.path.join(class_dir, 'eeg.mat'), {
            'data': np.array(trials_data),
            'labels': np.array(trials_labels),
            'fs': fs_eeg,
            'channels': n_channels,
            'class_name': class_name
        })
        
        # 创建对应的fNIRS数据
        fs_fnirs = 10  # Hz
        fnirs_data = []
        for trial in range(20):
            t_fnirs = np.linspace(0, duration, int(fs_fnirs * duration))
            
            # 模拟血流动力学响应
            hbo = np.zeros(len(t_fnirs))
            hbr = np.zeros(len(t_fnirs))
            
            # 延迟的激活模式
            activation_start = int(2.0 * fs_fnirs)  # 2秒延迟
            activation_end = int(5.0 * fs_fnirs)   # 持续3秒
            
            hbo[activation_start:activation_end] += np.random.normal(2.0, 0.5, activation_end - activation_start)
            hbr[activation_start:activation_end] += np.random.normal(-1.0, 0.3, activation_end - activation_start)
            
            fnirs_data.append(np.stack([hbo, hbr], axis=0))  # [2, time]
        
        savemat(os.path.join(class_dir, 'fnirs.mat'), {
            'data': np.array(fnirs_data),
            'fs': fs_fnirs,
            'channels': 2,  # HBO and HBR
            'class_name': class_name
        })
        
        print(f"  ✅ {class_name} 类别数据创建完成")
    
    print("✅ 增强样本数据创建完成!")
    return True

def main():
    print("🚀 BCI数据下载管理器")
    print("=" * 50)
    
    # 确保数据目录存在
    os.makedirs("data/raw", exist_ok=True)
    
    options = """
请选择数据获取方式:
1. 下载PhysioNet数据集 (需要网络连接)
2. 创建增强样本数据 (离线，推荐测试用)
3. 两者都执行
4. 仅显示当前数据状态
"""
    
    try:
        choice = input(options + "\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            download_physionet_data()
        elif choice == "2":
            create_realistic_sample_data()
        elif choice == "3":
            download_physionet_data()
            create_realistic_sample_data()
        elif choice == "4":
            pass
        else:
            print("❌ 无效选择")
            return
        
        # 显示数据状态
        print("\n📊 当前数据状态:")
        data_dir = "data/raw"
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                    for subitem in os.listdir(item_path):
                        if subitem.endswith('.mat'):
                            print(f"    📄 {subitem}")
        
        print("\n✅ 数据准备完成!")
        print("现在可以运行: python main.py --mode experiment")
        
    except KeyboardInterrupt:
        print("\n👋 用户取消操作")
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")

if __name__ == "__main__":
    main()