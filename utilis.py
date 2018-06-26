 #!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from glob import glob
import sys
from astropy.io import fits
import scipy.constants as const
from scipy import interpolate
import gc
import datetime
import urllib

import pandas as pd
#import lmfit
#from lmfit.models import GaussianModel
from harps.peakdetect import pkd
import xarray as xr
from joblib import Parallel,delayed
import h5py

from scipy.special import erf,erfc
from scipy.linalg import svd
from scipy.optimize import curve_fit, fsolve, newton, brentq
from scipy.optimize import minimize, leastsq, least_squares, OptimizeWarning, fmin_ncg
from scipy.optimize._lsq.least_squares import prepare_bounds
from scipy import odr

from harps import functions as hf
from harps import settings as hs

## IMPORT ENVIRONMENT VARIABLES AND USE THEM FOR OUTPUT




class SelectionFrame(object):
    def __init__(self,xpoints,ypoints):
        self._x_data, self._y_data = [xpoints,ypoints]
        self.peaks                 = pkd.peakdet(self._y_data,self._x_data)
        self._x_peaks, self._y_peaks = self.peaks.x, self.peaks.y
        self.create_main_panel()
        self.draw_figure()
        self._is_pick_started = False
        self._picked_indices = None
        self._is_finished = False
        
    def create_main_panel(self):
        self.dpi    = 300
        self.fig    = plt.figure(figsize=(16,9))
        self.canvas = self.fig.canvas
        self.axes   = self.fig.add_axes([0.05,0.05,0.9,0.9])
        #self.toolbar = wxagg.NavigationToolbar2WxAgg(self.canvas)
        #self.vbox = wx.BoxSizer(wx.VERTICAL)
        #self.vbox.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        #self.vbox.AddSpacer(25)
        #self.vbox.Add(self.toolbar, 0, wx.EXPAND)
        #self.panel.SetSizer(self.vbox)
        #self.vbox.Fit(self)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    def draw_figure(self):
        self.axes.clear()
        
        self.axes.plot(self._x_data, self._y_data, picker=False)
        self.axes.scatter(self._x_peaks, self._y_peaks, c='r', lw=0., picker=2)
        
        self.canvas.draw()
    def on_exit(self, event):
        self.Destroy()

    def picked_points(self):
        if self._picked_indices is None:
            return None
        else:
            return [ [self._x_data[i], self._y_data[i]]
                    for i in self._picked_indices ]

    def on_pick(self, event):
        if not self._is_pick_started:
            self._picked_indices = []
            self._is_pick_started = True

        for index in event.ind:
            if index not in self._picked_indices:
                self._picked_indices.append(index)
                self.axes.scatter(self._x_peaks[index], self._y_peaks[index], c='r', m='d',lw=0.)
                self.canvas.draw_idle()
        print(self.picked_points())

    def on_key(self, event):
        """If the user presses the Escape key then stop picking points and
        reset the list of picked points."""
        if 'r' == event.key:
            self._is_pick_started = False
            self._picked_indices = None
        if 'enter' == event.key:
            print("Selection done")
            self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('pick_event', self.on_pick))
            self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('key_press_event', self.on_key))
        if 'escape' == event.key:
            self._is_finished=True
        return
    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.fig.canvas.mpl_connect('key_press_event', self.press))
    def pick_lambda(self):
        ll = np.zeros(shape=(np.size(self._picked_points)))
        for i in range(ll.size):
            l = input("{}, lambda = ",format(i))
            ll[i] = l
        self._picked_lambda = ll
        return
