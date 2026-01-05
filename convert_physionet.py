#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将PhysioNet EDF数据转换为项目所需的MAT格式
"""

import os
import numpy as np
from scipy.io import savemat
import mne
from pathlib import Path

def convert_physionet_to_mat():
    """转换PhysioNet数据为项目格式"""
    
    print("🔄 转换PhysioNet EDF数据为MAT格式...")
    
    # 检查是否有PhysioNet数据
    physionet_dir = Path("data/raw/physionet")
    if not physionet_dir.exists():
        print("❌ 未找到PhysioNet数据目录")
        print("📥 尝试下载数据...")
        try:
            paths = mne.datasets.eegbci.load_data(subjects=[1, 2, 3], runs=[4, 8, 12], path='data/raw/physionet')
            print(f"✅ 下载完成: {len(paths)} 个文件")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            return False
    
    # 查找所有EDF文件
    edf_files = list(physionet_dir.rglob("*.edf"))
    print(f"📊 找到 {len(edf_files)} 个EDF文件")
    
    if len(edf_files) == 0:
        print("❌ 未找到EDF文件，创建模拟数据...")
        return create_simulated_physionet_data()
    
    # 转换每个受试者的数据
    subject_data = {}
    
    for edf_file in edf_files:
        # 解析文件名获取受试者和运行信息
        filename = edf_file.stem
        print(f"  📝 处理文件: {filename}")
        
        try:
            # 读取EDF文件
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # 提取数据
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            ch_names = raw.info['ch_names']
            
            # 确定运动想象类别
            if 'R04' in filename or 'T04' in filename:
                task_label = 0  # 左手
            elif 'R08' in filename or 'T08' in filename:
                task_label = 1  # 右手
            elif 'R12' in filename or 'T12' in filename:
                task_label = 2  # 脚
            else:
                task_label = 0  # 默认
            
            # 提取受试者编号
            subject_num = filename.split('R')[0].split('S')[1]
            subject_id = int(subject_num)
            
            if subject_id not in subject_data:
                subject_data[subject_id] = {
                    'eeg_data': [],
                    'labels': [],
                    'fnirs_data': [],
                    'events': []
                }
            
            # 添加EEG数据（取前64通道）
            eeg_channels = min(64, data.shape[0])
            subject_data[subject_id]['eeg_data'].append(data[:eeg_channels, :])
            subject_data[subject_id]['labels'].append(task_label)
            
            # 创建模拟fNIRS数据（因为PhysioNet主要是EEG）
            fnirs_samples = int(data.shape[1] / 100)  # fNIRS采样率约10Hz
            fnirs_data = np.random.randn(2, fnirs_samples) * 0.1
            fnirs_data[0, fnirs_samples//4:fnirs_samples//2] = 2.0  # HBO激活
            fnirs_data[1, fnirs_samples//4:fnirs_samples//2] = -1.0  # HBR激活
            subject_data[subject_id]['fnirs_data'].append(fnirs_data)
            
            # 创建事件标记
            event_time = data.shape[1] // 2  # 数据中间点
            subject_data[subject_id]['events'].append([event_time, 0, task_label + 1])
            
        except Exception as e:
            print(f"    ❌ 转换失败: {e}")
            continue
    
    # 保存转换后的数据
    print("\n💾 保存转换后的数据...")
    
    for subject_id, data_dict in subject_data.items():
        subject_dir = f"data/raw/subject_{subject_id:02d}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # 合并所有运行的数据
        if data_dict['eeg_data']:
            # 拼接EEG数据
            combined_eeg = np.concatenate(data_dict['eeg_data'], axis=1)
            
            # 拼接标签
            combined_labels = np.array(data_dict['labels'])
            
            # 拼接fNIRS数据
            combined_fnirs = np.concatenate(data_dict['fnirs_data'], axis=1)
            
            # 合并事件
            combined_events = np.array(data_dict['events'])
            
            # 保存MAT文件
            savemat(os.path.join(subject_dir, 'eeg.mat'), {
                'eeg_data': combined_eeg,
                'labels': combined_labels,
                'fs': np.array([[int(sfreq)]])
            })
            
            savemat(os.path.join(subject_dir, 'fnirs.mat'), {
                'fnirs_data': combined_fnirs,
                'fs': np.array([[10]])
            })
            
            savemat(os.path.join(subject_dir, 'events.mat'), {
                'events': combined_events,
                'event_types': np.array(['left_hand', 'right_hand', 'feet', 'tongue'])
            })
            
            print(f"  ✅ 受试者 {subject_id:02d}:")
            print(f"    EEG: {combined_eeg.shape}")
            print(f"    fNIRS: {combined_fnirs.shape}")
            print(f"    Labels: {combined_labels.shape}")
            print(f"    Events: {combined_events.shape}")
    
    print(f"\n🎉 数据转换完成!")
    print(f"📊 转换了 {len(subject_data)} 个受试者")
    print("🚀 现在可以运行: python main.py --mode experiment")
    
    return True

def create_simulated_physionet_data():
    """如果没有真实数据，创建模拟的PhysioNet风格数据"""
    
    print("🧠 创建模拟的PhysioNet风格数据...")
    
    for subject_id in range(1, 4):
        subject_dir = f"data/raw/subject_{subject_id:02d}"
        os.makedirs(subject_dir, exist_ok=True)
        
        # EEG数据 - 64通道，模拟PhysioNet格式
        duration = 60  # 60秒
        sfreq = 160  # PhysioNet常用采样率
        n_samples = duration * sfreq
        
        eeg_data = np.random.randn(64, n_samples) * 10  # 单位: μV
        
        # 添加alpha节律
        t = np.linspace(0, duration, n_samples)
        for ch in range(64):
            eeg_data[ch, :] += 5 * np.sin(2 * np.pi * 10 * t)  # 10Hz alpha
        
        # 运动想象任务标记
        labels = []
        events = []
        
        # 每15秒一个运动想象任务
        for i in range(4):
            task_start = i * 15 * sfreq
            task_end = (i * 15 + 3) * sfreq  # 3秒任务
            
            # 添加任务相关ERD/ERS
            if i == 0:  # 左手
                eeg_data[32:48, task_start:task_end] += 8  # 右半球激活
            elif i == 1:  # 右手
                eeg_data[16:32, task_start:task_end] += 8  # 左半球激活
            elif i == 2:  # 脚
                eeg_data[20:30, task_start:task_end] += 8  # 中央区激活
            else:  # 舌
                eeg_data[0:10, task_start:task_end] += 7  # 额叶激活
            
            labels.append(i)
            events.append([task_start, 0, i + 1])
        
        # 保存EEG数据
        savemat(os.path.join(subject_dir, 'eeg.mat'), {
            'eeg_data': eeg_data,
            'labels': np.array(labels),
            'fs': np.array([[sfreq]])
        })
        
        # fNIRS数据 - 模拟
        fnirs_samples = duration * 10  # 10Hz
        fnirs_data = np.random.randn(2, fnirs_samples) * 0.1
        
        # 添加与EEG同步的血流动力学响应
        for i in range(4):
            fnirs_start = i * 150  # 15秒 * 10Hz
            fnirs_end = fnirs_start + 60  # 6秒响应期
            
            if fnirs_end < fnirs_samples:
                fnirs_data[0, fnirs_start:fnirs_end] += 2.5  # HBO
                fnirs_data[1, fnirs_start:fnirs_end] -= 1.2  # HBR
        
        savemat(os.path.join(subject_dir, 'fnirs.mat'), {
            'fnirs_data': fnirs_data,
            'fs': np.array([[10]])
        })
        
        savemat(os.path.join(subject_dir, 'events.mat'), {
            'events': np.array(events),
            'event_types': np.array(['left_hand', 'right_hand', 'feet', 'tongue'])
        })
        
        print(f"  ✅ 受试者 {subject_id:02d}: 模拟PhysioNet数据创建完成")
    
    print("\n🎉 模拟PhysioNet数据创建完成!")
    print("📊 数据特点:")
    print("  - EEG: 64通道, 160Hz, 60秒")
    print("  - fNIRS: 2通道, 10Hz, 60秒") 
    print("  - 4类运动想象: 左手、右手、脚、舌")
    print("🚀 现在运行: python main.py --mode experiment")
    
    return True

if __name__ == "__main__":
    convert_physionet_to_mat()