import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp,fabs
from scipy import stats
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd 
import scipy.optimize as optimize
import scipy 

class ImpliedVolatility_Halley(object):
    def __init__(self, S, K, r, T,option_type):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        
    def bsmValue(self,sigma):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))
        d2 = d1 - sigma * sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:
            return self.S * stats.norm.cdf(d1) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2)

        elif self.optionType in ['Put', 'put', 'PUT']:
            return self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    ## Vega in BSM model (f')
    def bsmVega(self,sigma):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))
        vega = self.S * stats.norm.pdf(d1) * sqrt(self.T)
        return vega

    ## Vomma is BSM model (f")
    def bsmVomma(self,sigma):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)
        vomma = self.bsmVega(sigma) * (d1) * (d2) / sigma
        return vomma

    def get_implied_volatilities(self):
        func = lambda sigma: self.bsmValue(sigma) 
        fprime  = lambda sigma: self.bsmVega(sigma)
        fprime2 = lambda sigma: self.bsmVomma(sigma)

        impv = scipy.optimize.newton(f, 0.00,fprime, fprime2, xtol = 1e-12)[0] 
        return impv
   
