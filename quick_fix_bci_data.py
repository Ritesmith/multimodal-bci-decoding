#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速修复BCI数据格式，让项目可以立即运行
"""

import os
import numpy as np
from scipy.io import savemat

def quick_fix_data():
    """创建最小化但有效的BCI数据"""
    
    print("🔧 快速修复BCI数据格式...")
    
    # 只创建subject_01，用于快速测试
    subj_dir = "data/raw/subject_01"
    os.makedirs(subj_dir, exist_ok=True)
    
    # EEG数据: [channels, time] 
    eeg_data = np.random.randn(64, 70000) * 2  # 70秒连续信号
    
    savemat(os.path.join(subj_dir, 'eeg.mat'), {
        'eeg_data': eeg_data,
        'labels': np.array([0,1,2,3]),  # 简单标签
        'fs': np.array([[1000]])
    })
    
    # fNIRS数据: [channels, time]
    fnirs_data = np.random.randn(2, 700) * 0.1
    fnirs_data[0, 400:700] = 3.0  # HBO激活
    fnirs_data[1, 400:700] = -1.5  # HBR激活
    
    savemat(os.path.join(subj_dir, 'fnirs.mat'), {
        'fnirs_data': fnirs_data,
        'fs': np.array([[10]])
    })
    
    # 简单events
    events = np.array([
        [2000, 0, 1],  # 2秒时的事件
        [3000, 0, 2],  # 3秒时的事件  
        [4000, 0, 3],  # 4秒时的事件
        [5000, 0, 4],  # 5秒时的事件
    ], dtype=np.int32)
    
    savemat(os.path.join(subj_dir, 'events.mat'), {
        'events': events,
        'event_types': np.array(['left_hand', 'right_hand', 'feet', 'tongue'])
    })
    
    print("✅ BCI数据快速修复完成!")
    print("📁 数据位置: data/raw/subject_01/")
    print("🚀 现在运行: python main.py --mode experiment")
    print("\n💡 如需更多数据，请:")
    print("   1. 下载真实数据集 (推荐)")
    print("   2. 修改配置减少受试者数量")

if __name__ == "__main__":
    quick_fix_data()