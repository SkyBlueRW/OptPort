"""
协方差矩阵估计相关

sklearn中的Ledoit-Wolf Estimator的问题在于其shrink targe是相关性为0且等方差的矩阵，和事实情况不符

Ledoit & Wolf: Honey I shrunk the sample covariance matrix
Ledoit & Wolf: Improved estimation of the covariance matrix of stock returns with an application to portfolio selection
Newey & West: A simple positive semi-definite, heteroskedasticity and autocorrelation  consistent covariance matrix
"""

import numpy as np 
import pandas as pd 


def exp_cov_estimator(ret, halflife):
    """
    计算指数衰减的样本协方差矩阵
    """
    assert ret.isnull().sum().sum() == 0, "需要传入的ret无NaN,请自行根据需求进行填充"
    alpha = 1 - np.exp(-np.log(2) / halflife)
    
    label = ret.columns 

    return pd.DataFrame(
        np.cov(ret.T, aweights=[(1-alpha)**(i-1) for i in range(len(ret),0,-1)]),
        index=label, columns=label
    )



def sample_cov_estimator(ret, newey_west_lag=None):
    """
    计算(newey-west调整的)方差协方差矩阵

    Parameters
    ----------
    ret: pd.DataFrame
        date * name的df，通常为收益率，IC等
    newey_west_lag: int, optional
        newey-west调整的最大lag期数，若为None则不进行newey-west的调整
    
    Returns
    ------
    估计得到的方差协方差矩阵
    """
    assert ret.isnull().sum().sum() == 0, "需要传入的ret无NaN，请自行根据需求进行填充或drop"

    label = ret.columns
    ret = ret.values 

    # 数据维度
    T, n = ret.shape
    mean_ret = np.mean(ret, axis=0, keepdims=True)
    # 去均值
    ret = ret - mean_ret 
    # 计算sample covariance
    sample_cov = ret.transpose() @ ret / (T - 1)
    # 如果需要进行newey-west调整
    if newey_west_lag is not None:
        for q in range(1, (1+newey_west_lag)):
            temp_cov = (ret[q:].transpose() @ ret[:-q]) / (T - 1)
            w = 1 - q / (1 + newey_west_lag)
            sample_cov += w * (temp_cov + temp_cov.transpose())
    return pd.DataFrame(sample_cov, index=label, columns=label)


def constant_correlation_shrink(cov_matrix, shrink_intensity):
    """
    将方差协方差矩阵以线性的方式向固定相关性(所有相关性的均值)协方差矩阵shrink

    需要注意的是，cov_matrix中的varaince不应当存在0(constant variable)

    Parameters
    ---------
    cov_matrix: pd.DataFrame
        协方差矩阵，通常为sample covariance matrix
    shrink_intensity: float
        0 ~ 1的shrink强度

    Returns
    -------
    shrink 后的方差协方差矩阵
    """
    label = cov_matrix.index
    cov_matrix = cov_matrix.values
    n = len(cov_matrix)
    # 取出协方差矩阵中的方差
    var = np.diag(cov_matrix).reshape(-1, 1)
    # 标准差
    sqrt_var = var ** .5
    # 相关性为1时的协方差矩阵
    unit_cor_var = sqrt_var * sqrt_var.transpose()
    # 平均相关性
    average_cor = ((cov_matrix / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var
    # 回填方差
    np.fill_diagonal(prior, var)

    return pd.DataFrame(shrink_intensity * prior + (1 - shrink_intensity) * cov_matrix, index=label, columns=label)


def zero_correlation_shrink(cov_matrix, shrink_intensity, equal_variance=False):
    """
    将方差协方差矩阵以线性的方式向0相关性的协方差矩阵shrink
    SKlearn中提供的Ledoit Wolf Estimator类似这种方法

    Parameters
    ---------
    cov_matrix: pd.DataFrame
        协方差矩阵，通常为sample covariance matrix
    shrink_intensity: float
        0 ~ 1的shrink强度
    equal_variance: bool
        当为真时，prior的方差值相等为原协方差矩阵的均值
        若为否时，prior的方差值为远协方差矩阵的各自值
    
    Returns
    -------
    shrink 后的方差协方差矩阵
    """
    label = cov_matrix.index
    cov_matrix = cov_matrix.values
    # 使用各自的原方差 还是方差均值
    if not equal_variance:
        var = np.diag(cov_matrix)
    else:
        var = np.mean(np.diag(cov_matrix))
        var = np.array([var] * len(cov_matrix))
    
    # 相关性为0的prior协方差矩阵
    prior = np.diag(var)

    return pd.DataFrame(shrink_intensity * prior + (1 - shrink_intensity) * cov_matrix, index=label, columns=label)


def cal_ledoit_wolf_shrink_intensity(ret):
    """
    使得shrink estimator与真实cov 期望frobenius norm最小的 optimal shrink intensity
    Ref: Ledoit & Wolf: Honey I shrunk the sample covariance matrix

    Parameters
    ----------
    ret: pd.DataFrame
        收益率时间序列
    
    Returns
    -------

    """
    ret = ret.values
    t, n = ret.shape
    # demean
    mean_ret = np.mean(ret, axis=0, keepdims=True)
    ret -= mean_ret
    sample_cov = ret.transpose() @ ret / t 

    # cample average correlation
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = var ** .5
    unit_cor_var = sqrt_var * sqrt_var.transpose()
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var

    # pi-hat
    y = ret ** 2
    phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
    phi = phi_mat.sum() 

    # rho-hat
    theta_mat = ((ret ** 3).transpose() @ ret) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = (
        np.diag(phi_mat).sum()
        + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
    )

    # gamma-hat
    gamma = np.linalg.norm(sample_cov - prior, 'fro') ** 2

    # shrink constant
    kappa = (phi- rho) / gamma 
    shrink_intensity = max(0, min(1, kappa / t))

    return shrink_intensity
