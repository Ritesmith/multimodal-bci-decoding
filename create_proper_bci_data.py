#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建正确格式的BCI数据
"""

import os
import numpy as np
from scipy.io import savemat

def create_proper_bci_data():
    """创建符合项目要求的BCI数据格式"""
    
    print("🧠 创建正确格式的BCI数据...")
    
    # 创建多个受试者数据
    for subject_id in range(1, 4):  # 创建3个受试者
        print(f"  👤 创建受试者 {subject_id:02d}...")
        
        subject_dir = f"data/raw/subject_{subject_id:02d}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # EEG数据: [trials, channels, time] 格式
        n_trials = 120  # 120个trials
        n_channels = 64
        n_timepoints = 7000  # 7秒 * 1000Hz
        
        # 创建EEG数据
        eeg_data = np.random.randn(n_trials, n_channels, n_timepoints) * 2
        
        # 添加真实的EEG特征
        for trial in range(n_trials):
            t = np.linspace(0, 7, n_timepoints)
            
            # 添加alpha节律 (8-13Hz)
            alpha_freq = 10 + np.random.rand() * 3
            for ch in range(n_channels):
                eeg_data[trial, ch, :] += 1.5 * np.sin(2 * np.pi * alpha_freq * t)
            
            # 添加任务相关ERD/ERS (1-4秒)
            task_mask = (t >= 1.0) & (t <= 4.0)
            class_label = trial % 4  # 4类循环
            
            if class_label == 0:  # 左手
                for ch in range(32, 48):
                    eeg_data[trial, ch, task_mask] += 4 * np.sin(2 * np.pi * 20 * t[task_mask])
            elif class_label == 1:  # 右手
                for ch in range(16, 32):
                    eeg_data[trial, ch, task_mask] += 4 * np.sin(2 * np.pi * 20 * t[task_mask])
            elif class_label == 2:  # 脚
                for ch in range(20, 30):
                    eeg_data[trial, ch, task_mask] += 4 * np.sin(2 * np.pi * 25 * t[task_mask])
            else:  # 舌
                for ch in range(0, 10):
                    eeg_data[trial, ch, task_mask] += 3.5 * np.sin(2 * np.pi * 22 * t[task_mask])
        
        # 创建标签 [0, 1, 2, 3] 循环
        labels = np.tile([0, 1, 2, 3], n_trials // 4)
        
        # 保存EEG数据
        savemat(os.path.join(subject_dir, 'eeg.mat'), {
            'data': eeg_data,  # [120, 64, 7000]
            'labels': labels,  # [120]
            'fs': 1000,
            'n_channels': n_channels,
            'n_trials': n_trials
        })
        
        # fNIRS数据: [trials, channels, time] 格式
        fnirs_trials = 120
        fnirs_channels = 2  # HBO, HBR
        fnirs_timepoints = 70  # 7秒 * 10Hz
        
        fnirs_data = np.random.randn(fnirs_trials, fnirs_channels, fnirs_timepoints) * 0.1
        
        for trial in range(fnirs_trials):
            # 添加血流动力学响应 (4-7秒)
            hbo_response = np.zeros(fnirs_timepoints)
            hbr_response = np.zeros(fnirs_timepoints)
            
            # 延迟激活
            hbo_response[40:70] = 2.5  # 4-7秒
            hbr_response[40:70] = -1.2
            
            fnirs_data[trial, 0, :] = hbo_response + np.random.randn(fnirs_timepoints) * 0.2
            fnirs_data[trial, 1, :] = hbr_response + np.random.randn(fnirs_timepoints) * 0.1
        
        # 保存fNIRS数据
        savemat(os.path.join(subject_dir, 'fnirs.mat'), {
            'data': fnirs_data,  # [120, 2, 70]
            'fs': 10,
            'n_channels': fnirs_channels,
            'n_trials': fnirs_trials
        })
        
        # 创建事件标记
        events = []
        for i in range(n_trials):
            onset = 2.0 + i * 0.5  # 每0.5秒一个事件
            duration = 3.0
            event_type = (i % 4) + 1  # 1-4
            events.append([onset, duration, event_type])
        
        savemat(os.path.join(subject_dir, 'events.mat'), {
            'events': np.array(events),
            'class_names': ['left_hand', 'right_hand', 'feet', 'tongue']
        })
        
        print(f"    ✅ 受试者 {subject_id:02d}: {n_trials} trials")
    
    print("\n🎉 BCI数据创建完成!")
    print("📊 数据格式:")
    print("  - EEG: [trials, channels, time] = [120, 64, 7000]")
    print("  - fNIRS: [trials, channels, time] = [120, 2, 70]")
    print("  - 4个类别: 左手、右手、脚、舌头")
    print("  - 3个受试者: subject_01, subject_02, subject_03")
    print("\n🚀 现在运行: python main.py --mode experiment")

if __name__ == "__main__":
    create_proper_bci_data()