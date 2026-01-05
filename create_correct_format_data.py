#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建正确格式的BCI数据
符合 data_loader.py 的期望格式
"""

import os
import numpy as np
from scipy.io import savemat

def create_correct_format_data():
    """创建符合项目要求的BCI数据格式"""
    
    print("🧠 创建正确格式的BCI数据...")
    
    # 为3个受试者创建数据
    for subject_id in range(1, 4):
        print(f"  👤 创建受试者 {subject_id:02d}...")
        
        subject_dir = f"data/raw/subject_{subject_id:02d}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # EEG数据格式: [channels, time] 2D数组
        n_channels = 64
        duration = 7  # 秒
        fs = 1000  # Hz
        n_timepoints = duration * fs
        
        # 创建连续EEG信号 (而不是分trial的)
        eeg_data = np.random.randn(n_channels, n_timepoints * 120) * 2  # 120个trial的连续信号
        
        # 添加alpha节律
        t = np.linspace(0, duration * 120, n_timepoints * 120)
        alpha_freq = 10
        for ch in range(n_channels):
            eeg_data[ch, :] += 1.5 * np.sin(2 * np.pi * alpha_freq * t)
        
        # 标签数组
        labels = np.tile([0, 1, 2, 3], 30)  # 120个labels
        
        # 保存EEG数据 (格式: [channels, time])
        savemat(os.path.join(subject_dir, 'eeg.mat'), {
            'eeg_data': eeg_data,  # [64, 840000] - 2D数组
            'labels': labels,     # [120] - 1D数组  
            'fs': np.array([[fs]])  # [[1000]] - 2x2数组格式
        })
        
        # fNIRS数据: [channels, time] 2D数组
        fnirs_channels = 2
        fnirs_fs = 10  # Hz
        fnirs_timepoints = duration * fnirs_fs * 120  # 连续信号
        
        fnirs_data = np.random.randn(fnirs_channels, fnirs_timepoints) * 0.1
        
        # 添加周期性血流动力学响应
        trial_length = duration * fnirs_fs  # 70个时间点per trial
        for trial in range(120):
            start_idx = trial * trial_length
            end_idx = start_idx + trial_length
            
            # 每个trial的4-7秒激活期
            activation_start = start_idx + 40  # 4秒
            activation_end = start_idx + 70   # 7秒
            
            if activation_end <= fnirs_timepoints:
                fnirs_data[0, activation_start:activation_end] = 3.0  # HBO
                fnirs_data[1, activation_start:activation_end] = -1.5  # HBR
        
        # 保存fNIRS数据
        savemat(os.path.join(subject_dir, 'fnirs.mat'), {
            'fnirs_data': fnirs_data,  # [2, 8400] - 2D数组
            'fs': np.array([[fnirs_fs]])  # [[10]]
        })
        
        # 事件标记
        events = []
        for i in range(120):
            onset = 2.0 + i * duration  # 每7秒一个事件
            duration = 3.0
            event_type = (i % 4) + 1  # 1-4
            events.append([onset, duration, event_type])
        
        savemat(os.path.join(subject_dir, 'events.mat'), {
            'events': np.array(events),
            'class_names': ['left_hand', 'right_hand', 'feet', 'tongue']
        })
        
        print(f"    ✅ 受试者 {subject_id:02d} 完成")
        print(f"      EEG: {eeg_data.shape}")
        print(f"      fNIRS: {fnirs_data.shape}")
    
    print("\n🎉 正确格式的BCI数据创建完成!")
    print("📊 数据格式说明:")
    print("  - EEG: eeg_data[channels, time] = [64, 840000]")
    print("  - fNIRS: fnirs_data[channels, time] = [2, 8400]")
    print("  - Labels: [120]")
    print("  - Events: [120, 3]")
    print("\n🚀 现在运行: python main.py --mode experiment")

if __name__ == "__main__":
    create_correct_format_data()