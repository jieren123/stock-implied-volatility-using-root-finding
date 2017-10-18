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

class ImpliedVolatilityModel_MullerBisection(object):
    def __init__(self, S, K, r, T, option_type):
        self.S = S
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        #self.market_opt_prices = market_opt_prices
        #self.get_implied_volatilities = []

    def bsmValue(self, sigma):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))
        d2 = d1 - sigma * sqrt(self.T)

        if self.optionType in ['Call', 'call', 'CALL']:
            return self.S * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2,0.0, 1.0)

        elif self.optionType in ['Put', 'put', 'PUT']:
            return self.K * exp(-self.r * self.T) *(1 - stats.norm.cdf(d2,0.0, 1.0)) - self.S * (1 - stats.norm.cdf(d1, 0.0, 1.0))

        else:
            raise TypeError('the option_type argument must be either "call" or "put"')

    def swap_points(x):
        s = []
        s = x
        s.sort()
        f = s[1]
        sn = s[2]
        t = s[0]
        s[0] = f
        s[1] = sn
        s[2] = t
        return s

    def mullers_method(func, a, b, r, max_steps=MaxSteps):
        x = [a,b,r]
        for loopCount in range(max_steps):
            x = swap_points(x)
            y = func(x[0]), func(x[1]), func(x[2])
            h1 = x[1]-x[0]
            h2 = x[0]-x[2]
            lam = h2/h1
            c = y[0]
            a = (lam*y[1] - y[0]*((1.0+lam))+y[2])/(lam*h1**2.0*(1+lam))
            b = (y[1] - y[0] - a*((h1)**2.0))/(h1)
            if b > 0:
                root = x[0] - ((2.0*c)/(b+ (b**2 - 4.0*a*c)**0.5))
            else:
                root = x[0] - ((2.0*c)/(b- (b**2 - 4.0*a*c)**0.5))
            print "a = %.5f b = %.5f c = %.5f root = %.5f " % (a,b,c,root)
            print "Current approximation is %.9f" % root
            if abs(func(root)) > x[0]:
                x = [x[1],x[0],root]
            else:
                x = [x[2],x[0],root]
            x = swap_points(x)

    def get_implied_volatilities(self):
        f = lambda sigma: self.bsmValue(sigma) #- market_opt_prices[i]
        impv = self.mullers_method(f, 0.00, 0.9, 1.84)[0] # convergence to 1.84
