#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成真实BCI运动想象数据
"""

import os
import numpy as np
from scipy.io import savemat

def generate_realistic_bci_data():
    """生成真实感的BCI数据"""
    
    base_dir = 'data/raw/realistic_data'
    os.makedirs(base_dir, exist_ok=True)
    
    print('🧠 创建真实BCI运动想象数据...')
    
    # 4类运动想象，每类30个trial
    classes = ['left_hand', 'right_hand', 'feet', 'tongue']
    class_descriptions = ['左手运动想象', '右手运动想象', '脚部运动想象', '舌头运动想象']
    
    # 为每个受试者创建数据
    for subject_id in range(1, 4):  # 创建3个受试者
        subject_dir = os.path.join(base_dir, f'subject_{subject_id:02d}')
        os.makedirs(subject_dir, exist_ok=True)
        
        print(f'  👤 创建受试者 {subject_id:02d} 数据...')
        
        all_eeg_data = []
        all_labels = []
        all_fnirs_data = []
        
        for class_idx, (class_name, desc) in enumerate(zip(classes, class_descriptions)):
            print(f'    📝 {desc}...')
            
            # EEG数据参数
            fs_eeg = 1000
            duration = 7
            n_channels = 64
            n_trials = 30
            
            eeg_trials = []
            fnirs_trials = []
            
            for trial in range(n_trials):
                # 时间轴
                t = np.linspace(0, duration, int(fs_eeg * duration))
                
                # 基础EEG信号 (噪声 + 背景节律)
                eeg = np.random.randn(n_channels, len(t)) * 2
                
                # 添加alpha节律 (8-13Hz)
                alpha_freq = 10 + np.random.rand() * 3
                eeg += 1.5 * np.sin(2 * np.pi * alpha_freq * t)
                
                # 添加类别特定ERD/ERS模式 (1-4秒)
                task_mask = (t >= 1.0) & (t <= 4.0)
                
                if class_idx == 0:  # 左手 - 右半球激活
                    eeg[32:48, task_mask] += 4 * np.sin(2 * np.pi * 20 * t[task_mask])
                elif class_idx == 1:  # 右手 - 左半球激活  
                    eeg[16:32, task_mask] += 4 * np.sin(2 * np.pi * 20 * t[task_mask])
                elif class_idx == 2:  # 脚 - 中央区激活
                    eeg[20:30, task_mask] += 4 * np.sin(2 * np.pi * 25 * t[task_mask])
                else:  # 舌 - 额叶激活
                    eeg[0:10, task_mask] += 3.5 * np.sin(2 * np.pi * 22 * t[task_mask])
                
                # 添加眼电伪迹
                if np.random.rand() < 0.3:  # 30%概率有眨眼
                    blink_time = int(np.random.rand() * 2000)
                    blink_end = min(blink_time + 200, len(t))
                    eeg[60:62, blink_time:blink_end] += 5
                
                eeg_trials.append(eeg)
                
                # fNIRS数据 (血流动力学响应)
                fs_fnirs = 10
                t_fnirs = np.linspace(0, duration, int(fs_fnirs * duration))
                fnirs = np.zeros((2, len(t_fnirs)))  # HBO, HBR
                
                # 延迟的血流动力学响应 (4-7秒)
                hbo_mask = (t_fnirs >= 4.0) & (t_fnirs <= 7.0)
                fnirs[0, hbo_mask] += 2.5 + np.random.randn(np.sum(hbo_mask)) * 0.3  # HBO
                fnirs[1, hbo_mask] -= 1.2 + np.random.randn(np.sum(hbo_mask)) * 0.2  # HBR
                
                # 添加生理噪声
                fnirs += 0.1 * np.random.randn(2, len(t_fnirs))
                fnirs_trials.append(fnirs)
            
            all_eeg_data.extend(eeg_trials)
            all_labels.extend([class_idx] * n_trials)
            all_fnirs_data.extend(fnirs_trials)
        
        # 保存受试者数据
        savemat(os.path.join(subject_dir, 'eeg.mat'), {
            'data': np.array(all_eeg_data),  # [120 trials, 64 channels, 7000 timepoints]
            'labels': np.array(all_labels),
            'fs': fs_eeg,
            'n_channels': n_channels,
            'n_trials': len(all_eeg_data),
            'classes': classes
        })
        
        savemat(os.path.join(subject_dir, 'fnirs.mat'), {
            'data': np.array(all_fnirs_data),  # [120 trials, 2 channels, 70 timepoints]
            'fs': 10,
            'n_channels': 2,
            'n_trials': len(all_fnirs_data)
        })
        
        # 事件标记
        events = []
        for trial_idx in range(len(all_labels)):
            onset = 2.0 + trial_idx * 0.5  # 每0.5秒一个trial
            duration = 3.0
            event_type = all_labels[trial_idx] + 1  # 1-4
            events.append([onset, duration, event_type])
        
        savemat(os.path.join(subject_dir, 'events.mat'), {
            'events': np.array(events),
            'class_names': classes,
            'n_trials': len(events)
        })
        
        print(f'    ✅ 受试者 {subject_id:02d}: {len(all_eeg_data)} trials')
    
    print('\n🎉 真实BCI数据生成完成!')
    print(f'📊 数据统计:')
    print(f'  - 受试者数量: 3')
    print(f'  - 类别数量: 4 ({', '.join(classes)})')
    print(f'  - 每受试者trials: 120 (每类30)')
    print(f'  - EEG: 64通道, 1000Hz')
    print(f'  - fNIRS: 2通道, 10Hz')
    print(f'  - 保存位置: {base_dir}/')
    print('\n🚀 现在可以运行: python main.py --mode experiment')
    
    return True

if __name__ == "__main__":
    generate_realistic_bci_data()