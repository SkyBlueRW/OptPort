"""

"""

import numpy as np 
import pandas as pd 


def simple_minvar(A, b, cov):
    """
    给定约束的最小方差组合解析解
    Min x^T cov x
        s.t Ax = b

    Parameters
    ----------
    A: pd.DataFrame
        sid * char
    b: pd.Series, np.ndarray
        右侧约束
    cov: pd.DataFrame
        协方差矩阵
    
    """
    b = b.reindex(A.columns).values if isinstance(b, pd.Series) else b 
    z = np.linalg.solve(cov, A)
    res = np.dot(z, np.linalg.solve(A.T.dot(z), b))
    return pd.Series(res, index=A.index)


def shrink_u_cov(u, cov, bench, leverage):
    """
    当leverage更小时，投资组合会更加的偏离业绩基准(通过expected return shrink来实现)
    leverage = IR ** 2 / 6
    当leverage很小时，risk forced to be small so small industry active weight will be generated
    We could also use large leverage to maximize leverage but not too much risk 

    同样,standardize cov会帮助保持股票个数的稳定
    """
    # 求解出dollar neutral的最大sharpe组合
    A = u.to_frame("expected_return")
    A['const'] = 1.
    b = pd.Series([1, 0], index=['expected_return', 'const'])
    optl = simple_minvar(A, b, cov)
    # 求该组合对应的方差
    minvar = optl.dot(cov).dot(optl)
    
    # 标准化协方差矩阵
    cov = cov / abs(minvar) / 6
    # shrink u
    u = u * leverage + cov.dot(bench)

    return u, cov