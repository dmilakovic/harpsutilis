#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:05:36 2020

@author: dmilakov
"""

#------------------------------------------------------------------------------
# 
#                           D E C O R A T O R S
#
#------------------------------------------------------------------------------

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:            
            memo[x] = f(x)
        return memo[x]
    return helper