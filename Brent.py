import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd 
import scipy.optimize as optimize
import scipy 

class ImpliedVolatility_Brent(object):
    def __init__(self, S, K, r, T, option_type):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type

    def bsmValue(self, sigma):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))
        d2 = d1 - sigma * sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:
            return self.S * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2,0.0, 1.0)

        elif self.optionType in ['Put', 'put', 'PUT']:
            return self.K * exp(-self.r * self.T) *(1 - stats.norm.cdf(d2,0.0, 1.0)) - self.S * (1 - stats.norm.cdf(d1, 0.0, 1.0))

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    def get_implied_volatilities(self):
        f = lambda sigma: self.bsmValue(sigma) 
        impv = scipy.optimize.brentq(f, 0.00, 1.84, xtol = 1e-12)[0] # convergence to 1.84
        return impv

