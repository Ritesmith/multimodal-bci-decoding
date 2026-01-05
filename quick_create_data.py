#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速创建BCI测试数据
"""

import os
import numpy as np
from scipy.io import savemat

def quick_create_data():
    """快速创建符合格式的BCI数据"""
    
    print("🧠 快速创建BCI数据...")
    
    # 为3个受试者创建数据
    for subject_id in range(1, 4):
        print(f"  👤 受试者 {subject_id:02d}...")
        
        subject_dir = f"data/raw/subject_{subject_id:02d}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # EEG数据
        n_trials = 120
        n_channels = 64
        n_timepoints = 7000  # 7秒 * 1000Hz
        
        # 创建基础EEG数据
        eeg_data = np.random.randn(n_trials, n_channels, n_timepoints) * 2
        
        # 添加简单的类别特征
        for trial in range(n_trials):
            class_label = trial % 4  # 0-3类
            
            # 在每个trial的1-4秒添加类别特征
            start_idx = 1000  # 1秒
            end_idx = 4000    # 4秒
            
            if class_label == 0:  # 左手 - 右半球激活
                eeg_data[trial, 32:48, start_idx:end_idx] += 5
            elif class_label == 1:  # 右手 - 左半球激活
                eeg_data[trial, 16:32, start_idx:end_idx] += 5
            elif class_label == 2:  # 脚 - 中央区激活
                eeg_data[trial, 20:30, start_idx:end_idx] += 5
            else:  # 舌 - 额叶激活
                eeg_data[trial, 0:10, start_idx:end_idx] += 5
        
        # 创建标签
        labels = np.tile([0, 1, 2, 3], n_trials // 4)
        
        # 保存EEG数据
        savemat(os.path.join(subject_dir, 'eeg.mat'), {
            'data': eeg_data,
            'labels': labels,
            'fs': 1000,
            'n_channels': n_channels,
            'n_trials': n_trials
        })
        
        # fNIRS数据
        fnirs_data = np.random.randn(n_trials, 2, 70) * 0.1  # 2通道, 7秒*10Hz
        
        # 添加血流动力学响应
        for trial in range(n_trials):
            # 4-7秒延迟响应
            fnirs_data[trial, 0, 40:70] = 3.0  # HBO
            fnirs_data[trial, 1, 40:70] = -1.5  # HBR
        
        savemat(os.path.join(subject_dir, 'fnirs.mat'), {
            'data': fnirs_data,
            'fs': 10,
            'n_channels': 2,
            'n_trials': n_trials
        })
        
        # 事件标记
        events = []
        for i in range(n_trials):
            events.append([2.0 + i * 0.5, 3.0, (i % 4) + 1])
        
        savemat(os.path.join(subject_dir, 'events.mat'), {
            'events': np.array(events),
            'class_names': ['left_hand', 'right_hand', 'feet', 'tongue']
        })
        
        print(f"    ✅ 完成 {n_trials} trials")
    
    print("\n🎉 BCI数据创建成功!")
    print("📊 数据信息:")
    print("  - 3个受试者: subject_01, subject_02, subject_03")
    print("  - 每个受试者: 120 trials (4类运动想象)")
    print("  - EEG: 64通道, 1000Hz, 7秒")
    print("  - fNIRS: 2通道, 10Hz, 7秒")
    print("\n🚀 现在运行: python main.py --mode experiment")

if __name__ == "__main__":
    quick_create_data()