#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接下载并使用真实PhysioNet BCI数据
"""

import mne
import numpy as np
from scipy.io import savemat
import os

def download_and_setup_real_bci():
    """下载真实PhysioNet数据并设置项目使用"""
    
    print("🧠 下载真实PhysioNet BCI数据...")
    
    try:
        # 下载前3个受试者的运动想象数据
        print("📥 下载PhysioNet数据集...")
        subject_ids = [1, 2, 3]
        runs = [4, 8, 12]  # 左手、右手、脚运动想象
        
        raw_files = mne.datasets.eegbci.load_data(
            subjects=subject_ids,
            runs=runs,
            path='data/raw/physionet',
            update_path=True
        )
        
        print(f"✅ 下载成功: {len(raw_files)} 个文件")
        
        # 转换并保存为项目格式
        for i, subject_id in enumerate(subject_ids):
            print(f"🔄 转换受试者 {subject_id} 数据...")
            
            # 获取该受试者的所有运行
            subject_files = [f for f in raw_files if f"S0{subject_id:02d}" in f]
            
            all_eeg_data = []
            all_labels = []
            
            for j, raw_file in enumerate(subject_files):
                # 读取原始数据
                raw = mne.io.read_raw_edf(raw_file, preload=True, verbose=False)
                data = raw.get_data()
                sfreq = raw.info['sfreq']
                
                # 提取前64通道
                eeg_data = data[:64, :] if data.shape[0] >= 64 else data
                
                all_eeg_data.append(eeg_data)
                
                # 确定标签
                if 'R04' in raw_file or 'T04' in raw_file:
                    label = 0  # 左手
                elif 'R08' in raw_file or 'T08' in raw_file:
                    label = 1  # 右手
                elif 'R12' in raw_file or 'T12' in raw_file:
                    label = 2  # 脚
                else:
                    label = 3  # 舌（如果有的话）
                
                all_labels.extend([label] * 30)  # 假设30个trial
            
            # 合并所有运行的数据
            combined_eeg = np.concatenate(all_eeg_data, axis=1)
            combined_labels = np.array(all_labels[:len(all_eeg_data)])
            
            # 创建模拟fNIRS数据（因为PhysioNet主要是EEG）
            fnirs_duration = combined_eeg.shape[1] / sfreq
            fnirs_samples = int(fnirs_duration * 10)  # 10Hz
            fnirs_data = np.random.randn(2, fnirs_samples) * 0.1
            
            # 添加周期性激活
            for k in range(len(all_eeg_data)):
                start = k * (fnirs_samples // len(all_eeg_data))
                end = start + (fnirs_samples // len(all_eeg_data)) // 2
                if end < fnirs_samples:
                    fnirs_data[0, start:end] = 2.5  # HBO
                    fnirs_data[1, start:end] = -1.2  # HBR
            
            # 创建事件
            events = []
            for k in range(len(all_eeg_data)):
                start_time = k * (combined_eeg.shape[1] // len(all_eeg_data))
                events.append([start_time, 0, k + 1])
            
            # 保存为MAT格式
            subject_dir = f"data/raw/subject_{subject_id:02d}"
            os.makedirs(subject_dir, exist_ok=True)
            
            savemat(os.path.join(subject_dir, 'eeg.mat'), {
                'eeg_data': combined_eeg,
                'labels': combined_labels,
                'fs': np.array([[int(sfreq)]])
            })
            
            savemat(os.path.join(subject_dir, 'fnirs.mat'), {
                'fnirs_data': fnirs_data,
                'fs': np.array([[10]])
            })
            
            savemat(os.path.join(subject_dir, 'events.mat'), {
                'events': np.array(events, dtype=np.int32),
                'event_types': np.array(['left_hand', 'right_hand', 'feet', 'tongue'])
            })
            
            print(f"  ✅ 受试者 {subject_id}: EEG{combined_eeg.shape}, fNIRS{fnirs_data.shape}")
        
        print("\n🎉 真实BCI数据准备完成!")
        print("📊 数据特点:")
        print("  - 真实PhysioNet EEG信号")
        print("  - 模拟fNIRS信号（同步）")
        print("  - 3个受试者，多个运动想象任务")
        print("\n🚀 现在运行: python main.py --mode experiment")
        
        return True
        
    except ImportError:
        print("❌ MNE库未安装。请运行: pip install mne")
        return False
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_and_setup_real_bci()
    if success:
        print("\n🎯 任务完成！你现在有了真实的BCI数据。")
    else:
        print("\n💡 提示：你也可以使用已有的数据生成工具创建测试数据。")