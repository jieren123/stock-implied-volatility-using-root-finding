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

class ImpliedVolatility_Newton(object):
    def __init__(self, S, K, r, T, sigma,option_type,market_opt_prices,iter):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type
        self.market_opt_prices = market_opt_prices
        self.iter       = iter  # max iter

    def bsmValue(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T))
        d2 = d1 - self.sigma * sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:
            return self.S * stats.norm.cdf(d1) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2)
        elif self.optionType in ['Put', 'put', 'PUT']:
            return self.K * exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1)
        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    ## Vega in BSM model (f')
    def bsmVega(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * e.sqrt(self.T))
        vega = self.S * stats.norm.pdf(d1) * sqrt(self.T)
        return vega
    
    def bsmIVprediction(self):
        max_iter  = self.iter
        tolerance = 0.00000001
        for i in range(max_iter):
            f       = self.bsmValue() - self.market_opt_prices  # objective function
            f_prime = self.bsmVega()                # compute f_prime
            old_sigma  = self.sigma
            self.sigma = self.sigma - f/f_prime
            if (fabs(self.sigma - old_sigma) < tolerance):
                #print("total {:d}".format(i) + " iterations in newton method\n")
                return self.sigma

