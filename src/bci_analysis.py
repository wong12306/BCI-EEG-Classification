"""
BCI Competition IV 脑电信号分类
包含2a和2b数据集的处理
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from mne.time_frequency import psd_array_welch
from sklearn.metrics import confusion_matrix, classification_report

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class BCI2AAnalyzer:
    """2a数据集分析类（运动想象 + CSP）"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = None
        self.epochs = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """加载数据"""
        self.raw = mne.io.read_raw_gdf(
            self.file_path,
            preload=True,
            eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        )
        print("数据加载完成")
        print(self.raw.info)
        return self
    
    def preprocess(self, event_id=None, tmin=-2, tmax=3):
        """预处理：提取事件、创建epochs、滤波"""
        if event_id is None:
            event_id = {'eyesOpen': 6, 'eyesClosed': 7}
            
        events, event_dict = mne.events_from_annotations(self.raw)
        print("事件字典:", event_dict)
        
        # 创建epochs
        self.epochs = mne.Epochs(
            self.raw, events, event_id, 
            tmin=tmin, tmax=tmax,
            baseline=(tmin, 0),
            preload=True
        )
        
        # 带通滤波 7-30Hz
        self.epochs.filter(7, 30, method='iir')
        
        # 选择EEG通道
        eeg_channels = mne.pick_types(self.epochs.info, meg=False, eeg=True, exclude='bads')
        self.epochs.pick_channels([self.epochs.ch_names[i] for i in eeg_channels])
        
        # 提取数据
        self.X = self.epochs.get_data().astype(np.float64)
        self.y = self.epochs.events[:, -1]
        
        print(f"数据形状: {self.X.shape}, 标签形状: {self.y.shape}")
        return self
    
    def classify(self, n_splits=5):
        """分类评估"""
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        
        classifiers = {
            'SVM': SVC(kernel='linear', class_weight='balanced', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
            'LDA': LinearDiscrimriminantAnalysis()
        }
        
        results = {}
        
        for name, clf in classifiers.items():
            print(f"\n{'='*40}\n评估 {name}...\n{'='*40}")
            
            pipeline = Pipeline([
                ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ])
            
            try:
                scores = cross_val_score(pipeline, self.X, self.y, cv=cv, n_jobs=1)
                results[name] = scores
                print(f"{name} 平均准确率: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            except Exception as e:
                print(f"{name} 出错: {e}")
                
        self.results = results
        return results
    
    def plot_results(self, save_path=None):
        """可视化结果"""
        if not hasattr(self, 'results'):
            print("请先运行classify()")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 箱线图
        data_to_plot = [self.results[name] for name in self.results if self.results[name] is not None]
        labels = [name for name in self.results if self.results[name] is not None]
        axes[0].boxplot(data_to_plot, labels=labels)
        axes[0].set_title('不同分类器的交叉验证准确率')
        axes[0].set_ylabel('准确率')
        axes[0].grid(True)
        
        # 条形图
        means = [np.mean(self.results[name]) for name in labels]
        stds = [np.std(self.results[name]) for name in labels]
        axes[1].bar(labels, means, yerr=stds, capsize=5, color='skyblue')
        axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='随机猜测')
        axes[1].set_title('平均准确率对比')
        axes[1].set_ylabel('准确率')
        axes[1].legend()
        axes[1].grid(axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"图片已保存: {save_path}")
        
        plt.show()


class BCI2BAnalyzer:
    """2b数据集分析类（眼动事件 + PSD + 数据增强）"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = None
        self.epochs = None
        
    def load_and_preprocess(self):
        """加载和预处理"""
        self.raw = mne.io.read_raw_gdf(self.file_path, preload=True)
        
        # 设置通道类型
        self.raw.set_channel_types({
            'EOG:ch01': 'eog',
            'EOG:ch02': 'eog',
            'EOG:ch03': 'eog',
            'EEG:C3': 'eeg',
            'EEG:Cz': 'eeg',
            'EEG:C4': 'eeg'
        })
        
        # 提取事件
        events, eventnum = mne.events_from_annotations(self.raw)
        event_id = {
            'rejected': 1,
            'horizonEyeMove': 2,
            'verticalEyeMove': 3,
            'eyeRotation': 4,
            'eyeBlinks': 5
        }
        
        # 创建epochs
        self.epochs = mne.Epochs(
            self.raw, events, event_id, 
            tmax=3, event_repeated='merge', preload=True
        )
        
        # 滤波
        self.epochs = self.epochs.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        
        # 选择通道和事件
        channels = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03', 'EEG:C3', 'EEG:Cz', 'EEG:C4']
        train_data = self.epochs['eyeRotation', 'eyeBlinks'].get_data(picks=channels)
        train_labels = self.epochs['eyeRotation', 'eyeBlinks'].events[:, -1]
        
        print("原始数据形状:", train_data.shape)
        print("标签形状:", train_labels.shape)
        
        return train_data, train_labels
    
    @staticmethod
    def add_gaussian_noise(X, mean=0.0, std=0.05):
        """添加高斯噪声"""
        noise = np.random.normal(mean, std, X.shape)
        return X + noise
    
    @staticmethod
    def time_shift(X, max_shift=10):
        """时间平移"""
        X_shifted = []
        for trial in X:
            shift = np.random.randint(-max_shift, max_shift)
            shifted = np.roll(trial, shift, axis=-1)
            X_shifted.append(shifted)
        return np.array(X_shifted)
    
    def augment_data(self, X, y, noise_std=0.05, max_shift=10):
        """数据增强：3倍数据量"""
        # 高斯噪声
        X_noise = self.add_gaussian_noise(X, std=noise_std)
        
        # 时间平移
        X_shifted = self.time_shift(X, max_shift=max_shift)
        
        # 合并
        X_aug = np.concatenate([X, X_noise, X_shifted], axis=0)
        y_aug = np.concatenate([y, y, y], axis=0)
        
        print("增强后样本数:", X_aug.shape[0])
        return X_aug, y_aug
    
    @staticmethod
    def extract_psd_features(data, sfreq=250, fmin=7, fmax=30):
        """提取PSD特征"""
        psd_features = []
        for trial in data:
            psd, freqs = psd_array_welch(
                trial, sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False
            )
            psd_features.append(psd.flatten())
        return np.array(psd_features)
    
    def classify(self, X, y, use_augmentation=True):
        """分类评估"""
        if use_augmentation:
            X, y = self.augment_data(X, y)
            
        # 提取PSD特征
        X_psd = self.extract_psd_features(X)
        print("PSD特征形状:", X_psd.shape)
        
        # 定义模型
        models = {
            'SVM': SVC(kernel='rbf'),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'LDA': LinearDiscriminantAnalysis()
        }
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', model)
            ])
            scores = cross_val_score(pipeline, X_psd, y, cv=cv, scoring='accuracy')
            results[name] = scores
            print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
            
        self.results = results
        return results
    
    def plot_results(self, save_path=None):
        """可视化"""
        if not hasattr(self, 'results'):
            print("请先运行classify()")
            return
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[self.results[k] for k in self.results.keys()])
        plt.xticks(ticks=range(len(self.results)), labels=self.results.keys(), fontsize=12)
        plt.ylabel("准确率", fontsize=13)
        plt.title("基于PSD特征的分类器准确率比较", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"图片已保存: {save_path}")
            
        plt.show()


# 便捷函数
def run_2a_analysis(file_path, plot=True):
    """运行2a数据集完整分析"""
    analyzer = BCI2AAnalyzer(file_path)
    analyzer.load_data().preprocess()
    results = analyzer.classify()
    if plot:
        analyzer.plot_results('2a_results.png')
    return results

def run_2b_analysis(file_path, plot=True):
    """运行2b数据集完整分析"""
    analyzer = BCI2BAnalyzer(file_path)
    X, y = analyzer.load_and_preprocess()
    results = analyzer.classify(X, y, use_augmentation=True)
    if plot:
        analyzer.plot_results('2b_results.png')
    return results