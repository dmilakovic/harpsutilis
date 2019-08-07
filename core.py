#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:15 2018

@author: dmilakov
"""
# system 
import gc
#import tqdm
import argparse

import sys
import warnings
import os
import time
from glob import glob
import json


# numerical 
import numpy as np
import pandas as pd
import numbers
#import xarray as xr

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
#import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt