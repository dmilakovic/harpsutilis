#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:03:29 2023

@author: dmilakov
"""
import sys
import time
import logging
#------------------------------------------------------------------------------
#
#                           P R O G R E S S   B A R 
#
#------------------------------------------------------------------------------
def update(progress,name=None,time=None,logger=None):
    '''
    

    Parameters
    ----------
    progress : TYPE
        DESCRIPTION.
    name : TYPE, optional
        DESCRIPTION. The default is None.
    time : TYPE, optional
        Elapsed time (in seconds). The default is None.
    logger : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    # https://stackoverflow.com/questions/3160699/python-progress-bar
    barLength = 40 
    status = ""
    name = name if name is not None else ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt\r\n"
    if progress >= 1:
        progress = 1
        status = "Done\r\n"
    block = int(round(barLength*progress))
    mess  = (name,"#"*block + "-"*(barLength-block), progress*100, status)
    text = "Progress [{0}]: [{1}] {2:8.3f}% {3}".format(*mess)
    if time is not None:
        h, m, s = get_time(time)
        text = text + f"  elapsed time: {h:02d}h {m:02d}m {s:02d}s"
    if logger is not None:
        logger.info(text)
    else:
        sys.stdout.write("\r"+text)
        sys.stdout.flush()
        
def get_time(worktime):
    """
    Returns the work time in hours, minutes, seconds

    Outputs:
    --------
           h : hour
           m : minute
           s : second
    """					
    m,s = divmod(worktime, 60)
    h,m = divmod(m, 60)
    h,m,s = [int(value) for value in (h,m,s)]
    return h,m,s        