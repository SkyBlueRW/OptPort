# OptPort 

OptPort is a package for portfolio optimization. It can be used for portfolio / signal optimization as well as related analytical component like constraint attribution, etc... 

The package uses conic optimization from mosek solver to deliever commonly used portfolio optimizations like:
    
    - Mean Variance
    - Max Sharpe
    - Risk Budeget
    - Risk Parity
    - Black-Litterman application

Estimation:
The package provides some of the basic estimator implementations for both expected return and variance/covariance matrix
    
    - variance coariance matrix: sample estimator, Ledoit & Wolf shrink estimator, Newey-West estimator, etc.
    
    - various linear shrinkage for expected return to control vol, leverage, etc..


Constraint Attribution





