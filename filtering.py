#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:10:05 2022

@author: dmilakov
"""
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib.pyplot as plt
import vputils.plot as hplt

def band(y,low_f,high_f,x=None):
    ''' 
    https://stackoverflow.com/questions/19122157/fft-bandpass-filter-in-python
    '''
    # FFT the signal
    sig_fft = rfft(y)
    # copy the FFT results
    sig_fft_filtered = sig_fft.copy()
    
    x = x if x is not None else np.arange(len(y))
    # obtain the frequencies using scipy function (assumes uniform sampling)
    freq = fftfreq(len(y), d=x[1]-x[0])
    # print(freq)
    # plt.plot(y)
    # fig,(ax1,ax2,ax3,ax4) = plt.subplots(2,2)
    fig = hplt.Figure(2,2,[3,1],figsize=(9,6))
    ax1,ax3,ax2,ax4 = (fig.ax() for i in range(4))
    ax1.plot(x,y,drawstyle='steps-mid')
    ax2.plot(freq,sig_fft,drawstyle='steps-mid')
    # [plt.axhline(f) for f in [low_f,high_f]]
    # high-pass filter by assign zeros to the 
    # FFT amplitudes where the absolute 
    # frequencies smaller than the cut-off 
    condition1 = np.abs(freq) > low_f
    condition2 = np.abs(freq) < high_f
    condition = condition1 & condition2
    print(np.sum(condition1), np.sum(condition2), np.sum(condition))
    sig_fft_filtered[~condition] = 0
    [ax2.axvline(f,c='k',ls=':') for f in [-low_f,-high_f,low_f,high_f]]
    ax2.plot(freq[condition],sig_fft_filtered[condition],drawstyle='steps-mid')
    # get the filtered signal in time domain
    filtered = irfft(sig_fft_filtered)
    ax1.plot(x,filtered,drawstyle='steps-mid')
    
    ax3.hist(y,bins=100)
    # ax3.hist(filtered,bins=100)
    return filtered

