# 脑电信号分类研究 - BCI Competition IV

基于BCI Competition IV数据集的运动想象与眼动事件脑电信号分类研究。

## 项目简介

本项目探讨数据预处理、特征提取（CSP/PSD）及分类模型在提高EEG分类性能方面的作用。

**实验结果：**
- 2a数据集（运动想象）：LDA模型最佳，准确率 **58.28%**
- 2b数据集（眼动事件）：KNN模型最佳，准确率 **61.11%**（使用数据增强）

## 环境要求

- Python 3.9+
- 主要依赖：mne, scikit-learn, numpy, matplotlib

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 2a数据集实验
python scripts/run_2a_experiment.py

# 2b数据集实验  
python scripts/run_2b_experiment.py
```

## 数据集

数据来自 [BCI Competition IV](http://www.bbci.de/competition/iv/)，需自行下载。