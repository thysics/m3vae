#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries:
import numpy as np
import scipy
import scipy.stats as st
from scipy.stats import cauchy
from scipy.stats import gamma
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn import mixture
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from matplotlib.ticker import ScalarFormatter, NullFormatter

def prob_silverman(data, k_peaks, widths, N_boots=500, x_eval_kde=None):
# Calculate probabilities for having more than k_peaks in the data set 
# (Probability for rejecting the null hypothesis H0 that there are k peaks)
# (original algorithm by Silverman 1981, i.e. just counting peaks for each bootstrap sample)
# Smoothed bootstrap procedure as in Silverman 1981
# Check expressions in Silverman (1981) and a pedagogical explanation in
# http://adereth.github.io/blog/2014/10/12/silvermans-mode-detection-method-explained/

# This test is known to be conservative, in the sense that it is not rigorous at rejecting models with k peaks.
# In other words, in some cases there might be more than k statistically significant peaks, but the test still does not reject models with k peaks
# On the other hand, if the test rejects a model with k peaks, you can bet that there is in fact evidence for more peaks.

    if (len(np.shape(k_peaks)) == 0):
        k_peaks = np.array([k_peaks])

    if (x_eval_kde is None):
        x_eval_kde = np.linspace(min(data), max(data), 1000)
    
    h_crit = np.zeros(len(k_peaks))
    print ('Determining h_crit...')
    for i in range(len(k_peaks)): # k is number of peaks
        h_crit[i] = crit_window_width(data, k_peaks[i], widths, x_eval_kde)
#             h_crit[i] = crit_window_width(data, k_peaks[i], widths)
        # print ("N. of peaks=",k_peaks[i], "h_crit =",np.round(h_crit[i],4))
    print ("h_crit:", np.round(h_crit, 4))
    if (np.any(np.isnan(h_crit))):
        print ('NaN entries found in h_crit. It seems it needs to change widths interval.')
        return

    sigma = np.std(data)
    print ('Dispersion of data:',np.round(sigma,4))
    P = np.zeros(len(k_peaks))
    print ("Now calculate probabilities...")
    N = len(data)
    
    for k in range(len(k_peaks)):
        h_k = h_crit[k]
        one_over_sqrt = (1./np.sqrt(1. + h_k**2/sigma**2))
        kd = KernelDensity(bandwidth=h_k, kernel='gaussian')
        
        print ("N. of peaks:",k_peaks[k],"h_crit=",np.round(h_k,4))
        N_peaks = np.zeros(N_boots)
        
        for i in range(N_boots):
            sampled_data = np.random.choice(data, size=N)
            eps = np.random.normal(0, 1, N)
            smooth_data = one_over_sqrt*(sampled_data + h_k*eps)

            kd.fit(smooth_data[:, None])
            logprob = kd.score_samples(x_eval_kde[:, None])
            kde = np.exp(logprob)
            peaks, _ = find_peaks(kde)
            N_peaks[i] = len(peaks)
        
        P[k] = len(N_peaks[N_peaks<=k_peaks[k]])/N_boots
        print ("P=",P[k])
    return P

def crit_window_width(x, k, widths, x_eval_kde=None):
    # x is the array with data
    # k is number of peaks for which we want the critical window width
    # widths is the array with window widths to use to find the critical one for each number of peaks
    # x_eval_kde has the positions where the KDE will be evaluated
    # It returns the critical window width, i.e. the minimum window width, for the kde to have at most k peaks. A smaller window produces mode than k peaks 
    
# Binary search to find critical, 
    if (x_eval_kde is None):
        x_eval_kde = np.linspace(min(x), max(x), 1000)
    lo = 0
    hi = len(widths)
    N_peaks = 0
    while lo <= hi:
        mid = (lo+hi)//2
        w = widths[mid]
        kd = KernelDensity(bandwidth=w, kernel='gaussian')
        kd.fit(x[:, None])
        logprob = kd.score_samples(x_eval_kde[:, None])
        kde = np.exp(logprob)
        peaks, _ = find_peaks(kde)
        N_peaks = len(peaks)
        if (lo==hi): break
        if (N_peaks <= k):
            hi = mid
            if hi==0: 
                print ("Minimum window width achieved. Set up smaller values")
                return
        else:
            lo = mid+1
            if lo==len(widths): 
                print ("Maximum window width achieved. Set up larger values")
                return
    return w

