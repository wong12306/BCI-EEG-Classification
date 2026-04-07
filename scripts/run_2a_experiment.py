#!/usr/bin/env python3
"""
运行2a数据集实验
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bci_analysis import run_2a_analysis

# 修改这里的路径为你的实际数据路径
DATA_PATH = 'D:/240/MI_BCICIV_2a/BCICIV_2a_gdf/A01E.gdf'

if __name__ == '__main__':
    print("=" * 50)
    print("开始2a数据集实验（运动想象 + CSP）")
    print("=" * 50)
    
    results = run_2a_analysis(DATA_PATH)
    
    print("\n" + "=" * 50)
    print("实验完成！最佳模型：")
    best_model = max(results.items(), key=lambda x: x[1].mean())
    print(f"{best_model[0]}: {best_model[1].mean():.4f}")
    print("=" * 50)