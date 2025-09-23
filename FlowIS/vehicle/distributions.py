import torch
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class MarginalGaussianMixture():
    """可以实现对任意维度边缘分布进行采样的混合高斯"""
    def __init__(self, n_components=1, random_state=None):
        """初始化高斯混合模型"""
        self.n_components = n_components
        self.random_state = random_state
        self.weights = None      # 各高斯分布的权重
        self.means = None        # 各高斯分布的均值
        self.covariances = None  # 各高斯分布的协方差矩阵
        
    def fit(self, X):
        """使用 sklearn 的 GMM 拟合数据并提取参数"""
        # 拟合数据
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
            covariance_type='full'  # 完整协方差矩阵
        ).fit(X)
        
        # 提取参数
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_
        
        return self
    
    def sample(self, n_samples=1):
        """从混合分布中采样"""
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
            
        # 确定每个成分生成的样本数量
        n_samples_per_component = np.random.multinomial(n_samples, self.weights)
        
        # 为每个成分生成样本
        samples = []
        for i in range(self.n_components):
            if n_samples_per_component[i] > 0:
                # 从第 i 个高斯分布中采样
                component_samples = np.random.multivariate_normal(
                    mean=self.means[i],
                    cov=self.covariances[i],
                    size=n_samples_per_component[i]
                )
                samples.append(component_samples)
        
        # 合并所有样本并打乱顺序
        if samples:
            samples = np.vstack(samples)
            np.random.shuffle(samples)
            return samples
        else:
            return np.array([])
    
    def pdf(self, X):
        """计算概率密度函数值"""
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
            
        # 对每个样本，计算各成分的加权密度之和
        density = np.zeros(X.shape[0])
        for i in range(self.n_components):
            # 使用 scipy 的多元正态分布计算密度
            component_density = multivariate_normal.pdf(
                X, 
                mean=self.means[i], 
                cov=self.covariances[i],
                allow_singular=True  # 允许奇异协方差矩阵
            )
            density += self.weights[i] * component_density
            
        return density
        
    def sample_marginal(self, n_samples=1, dims=[0, 1]):
        """
        对指定维度的边缘分布进行采样
        
        参数:
            n_samples: 采样数量
            dims: 要保留的维度索引列表，默认为前两个维度 [0, 1]
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
            
        # 提取边缘分布的参数
        marginal_means = self.means[:, dims]
        marginal_covs = []
        
        for cov in self.covariances:
            # 提取指定维度的协方差矩阵子块
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)
            
        # 确定每个成分生成的样本数量
        n_samples_per_component = np.random.multinomial(n_samples, self.weights)
        
        # 为每个成分生成样本
        samples = []
        for i in range(self.n_components):
            if n_samples_per_component[i] > 0:
                # 从第 i 个高斯分布的边缘分布中采样
                component_samples = np.random.multivariate_normal(
                    mean=marginal_means[i],
                    cov=marginal_covs[i],
                    size=n_samples_per_component[i]
                )
                samples.append(component_samples)
        
        # 合并所有样本并打乱顺序
        if samples:
            samples = np.vstack(samples)
            np.random.shuffle(samples)
            return samples
        else:
            return np.array([])
    
    def logpdf(self, X):
        """计算对数概率密度函数值"""
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
            
        # 对每个样本，计算各成分的对数密度
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # 使用 scipy 的多元正态分布计算对数密度
            log_densities[:, i] = multivariate_normal.logpdf(
                X, 
                mean=self.means[i], 
                cov=self.covariances[i],
                allow_singular=True  # 允许奇异协方差矩阵
            )
        
        # 使用 logsumexp 函数稳定地计算对数加权和
        # log(sum_i w_i * exp(log_p_i)) = log(sum_i exp(log(w_i) + log_p_i))
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf
    
    def logpdf_marginal(self, X, dims=[0, 1]):
        """
        计算指定维度边缘分布的对数概率密度
        
        参数:
            X: 输入样本，形状为 (n_samples, len(dims))
            dims: 对应的维度索引列表
        """
        if self.weights is None:
            raise ValueError("模型尚未拟合，请先调用 fit 方法")
            
        # 提取边缘分布的参数
        marginal_means = self.means[:, dims]
        marginal_covs = []
        
        for cov in self.covariances:
            # 提取指定维度的协方差矩阵子块
            marginal_cov = cov[np.ix_(dims, dims)]
            marginal_covs.append(marginal_cov)
            
        # 对每个样本，计算各成分的对数密度
        log_densities = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            # 使用 scipy 的多元正态分布计算对数密度
            log_densities[:, i] = multivariate_normal.logpdf(
                X, 
                mean=marginal_means[i], 
                cov=marginal_covs[i],
                allow_singular=True  # 允许奇异协方差矩阵
            )
        
        # 使用 logsumexp 函数稳定地计算对数加权和
        log_weights = np.log(self.weights)
        log_pdf = np.logaddexp.reduce(log_weights + log_densities, axis=1)
        
        return log_pdf    

class NormalProposalDistribution:
    """
    正态分布提议分布，支持对数概率计算和采样
    """
    def __init__(self, mean=0.0, std=1.0):
        """
        初始化正态分布提议分布
        
        参数:
            mean: 分布均值
            std: 分布标准差（必须为正数）
        """
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.var = std ** 2  # 方差
        
        # 预计算常量以提高对数概率计算效率
        self.log_const = -0.5 * np.log(2 * np.pi) - np.log(self.std)
        
    def sample(self):
        """从正态分布中生成一个随机样本"""
        return np.array([[np.random.normal(self.mean, self.std)]])
    
    def logpdf(self, x):
        """
        计算样本x的对数概率密度
        
        参数:
            x: 样本值
            
        返回:
            对数概率密度值
        """
        # 正态分布的对数概率密度公式: 
        # log(p(x)) = -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²
        return self.log_const - 0.5 * ((x - self.mean) / self.std) ** 2
    
    def pdf(self, x):
        """
        计算样本x的概率密度
        
        参数:
            x: 样本值
            
        返回:
            概率密度值
        """
        return np.exp(self.logpdf(x)) 
    
class PiecewiseUniformDistribution:
    """
    分段均匀分布采样器，支持不同区间设置不同概率密度
    """
    def __init__(self, breakpoints, densities):
        """
        初始化分段均匀分布
        
        参数:
            breakpoints: 区间断点列表，例如 [0, 0.3, 0.6, 1.0]
            densities: 各区间的密度系数列表，例如 [18, 15, 8]
        """
        # 验证输入合法性
        if len(breakpoints) - 1 != len(densities):
            raise ValueError("断点数量必须比密度系数多1")
        
        if not np.all(np.diff(breakpoints) > 0):
            raise ValueError("断点必须严格递增")
        
        if any(d <= 0 for d in densities):
            raise ValueError("密度系数必须为正数")
        
        self.breakpoints = np.array(breakpoints)
        self.densities = np.array(densities)
        
        # 计算归一化常数 a
        intervals = np.diff(self.breakpoints)
        self.a = 1.0 / np.sum(intervals * self.densities)
        
        # 计算累积分布函数(CDF)的断点
        self.cdf_breakpoints = np.zeros(len(breakpoints))
        for i in range(1, len(breakpoints)):
            self.cdf_breakpoints[i] = self.cdf_breakpoints[i-1] + \
                                      intervals[i-1] * densities[i-1] * self.a
    
    def sample(self, size=1):
        """从分布中采样"""
        # 从均匀分布 U(0,1) 采样
        u = np.random.rand(size)
        
        # 初始化样本数组
        samples = np.zeros_like(u)
        
        # 对每个样本，根据 u 的值确定所属区间并应用逆变换
        for i in range(len(self.breakpoints) - 1):
            mask = (u >= self.cdf_breakpoints[i]) & (u < self.cdf_breakpoints[i+1])
            
            if np.any(mask):
                # 逆变换公式：x = (u - cdf[i]) / (density[i]*a) + breakpoint[i]
                samples[mask] = (u[mask] - self.cdf_breakpoints[i]) / \
                               (self.densities[i] * self.a) + self.breakpoints[i]
        
        return samples
    
    def log_prob(self, x):
        """计算对数概率密度"""
        # 初始化对数概率数组，默认值为 -inf（超出范围的点）
        log_probs = np.full_like(x, -np.inf, dtype=np.float64)
        
        # 对每个区间，检查 x 是否在其中，并设置对应的对数概率
        for i in range(len(self.breakpoints) - 1):
            mask = (x >= self.breakpoints[i]) & (x < self.breakpoints[i+1])
            
            # 特殊处理最后一个区间的右端点（闭区间）
            if i == len(self.breakpoints) - 2:
                mask = mask | (x == self.breakpoints[i+1])
            
            if np.any(mask):
                log_probs[mask] = np.log(self.densities[i] * self.a)
        
        return log_probs
    
    def prob(self, x):
        """计算概率密度"""
        return np.exp(self.log_prob(x))
