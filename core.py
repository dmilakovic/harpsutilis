#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:15 2018

@author: dmilakov
"""
# system 
import gc
import dill as pickle
import tqdm


import sys
import warnings
import os
import time
from glob import glob
import json
import logging

# numerical 
import numpy as np
import pandas as pd
import xarray as xr

# scientific computing / fitting
from scipy.optimize import curve_fit, leastsq
from scipy import odr, interpolate
from numpy.polynomial import Polynomial as P
from scipy.signal import welch

# multiprocessing
import multiprocessing as mp

# FITS file manipulation
from astropy.io import fits
from fitsio import FITS, FITSHDR

# plotting
import matplotlib.pyplot as plt