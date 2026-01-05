#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.io import savemat

def main():
    print("Creating BCI data...")
    
    # 确保目录存在
    for subj in range(1, 4):
        subj_dir = f"data/raw/subject_{subj:02d}"
        os.makedirs(subj_dir, exist_ok=True)
        print(f"Directory: {subj_dir}")
        
        # EEG数据
        eeg_data = np.random.randn(64, 840000) * 2
        labels = np.tile([0,1,2,3], 30)
        
        eeg_file = os.path.join(subj_dir, 'eeg.mat')
        savemat(eeg_file, {
            'eeg_data': eeg_data,
            'labels': labels,
            'fs': np.array([[1000]])
        })
        
        # fNIRS数据
        fnirs_data = np.random.randn(2, 8400) * 0.1
        for trial in range(120):
            start = trial * 70
            fnirs_data[0, start+40:start+70] = 3.0
            fnirs_data[1, start+40:start+70] = -1.5
        
        fnirs_file = os.path.join(subj_dir, 'fnirs.mat')
        savemat(fnirs_file, {
            'fnirs_data': fnirs_data,
            'fs': np.array([[10]])
        })
        
        # 事件文件
        events = [[2.0 + i*7.0, 3.0, (i%4)+1] for i in range(120)]
        events_file = os.path.join(subj_dir, 'events.mat')
        savemat(events_file, {
            'events': np.array(events)
        })
        
        print(f"  Files created: {eeg_file}, {fnirs_file}, {events_file}")
        print(f"  Sizes: {os.path.getsize(eeg_file)}, {os.path.getsize(fnirs_file)}, {os.path.getsize(events_file)}")
    
    print("\nAll BCI data created successfully!")

if __name__ == "__main__":
    main()