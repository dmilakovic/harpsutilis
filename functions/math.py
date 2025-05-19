#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:27:52 2023

@author: dmilakov
"""
import numpy as np
import math as mth
import scipy.interpolate as interpolate
from scipy.optimize import minimize, leastsq, curve_fit, brentq
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Sequence


from scipy.ndimage import gaussian_filter1d

#------------------------------------------------------------------------------
# 
#                           M A T H E M A T I C S
#
#------------------------------------------------------------------------------
def derivative1d(y,x=None,order=1,method='coeff'):
    ''' Get the first derivative of a 1d array y. 
    Modified from 
    http://stackoverflow.com/questions/18498457/
    numpy-gradient-function-and-numerical-derivatives'''
    # def _contains_nan(array):
    #     return np.any(np.isnan(array))
    # contains_nan = [_contains_nan(array) for array in [y,x]]
    # x = x if x is not None else np.arange(len(y))
    # if any(contains_nan)==True:
    #     return np.zeros_like(y)
    # else:
    #     pass
    x = x if x is not None else np.arange(len(y))
    if method=='forward':
        dx = np.diff(x)
        # dx  = np.append(dx_,np.zeros(order))
        dy = np.diff(y,order)
        # dy  = np.append(dy_,np.zeros(order))
        if order==2:
            d   = dy/dx[:-1]
        else:
            d = dy/dx
        d = np.append(d,np.zeros(order))
    if method == 'central':
        z1  = np.hstack((y[0], y[:-1]))
        z2  = np.hstack((y[1:], y[-1]))
        dx1 = np.hstack((0, np.diff(x)))
        dx2 = np.hstack((np.diff(x), 0))  
        if np.all(np.asarray(dx1+dx2)==0):
            dx1 = dx2 = np.ones_like(x)/2
        d   = (z2-z1) / (dx2+dx1)
    if method == 'coeff':
        d = derivative(y,x,order)
    return d
def derivative(y_axis,x_axis=None,order=1,accuracy=4):
    if order==1:
        _coeffs = {2:[-1/2,0,1/2],
                  4:[1/12,-2/3,0,2/3,-1/12],
                  6:[-1/60,3/20,-3/4,0,3/4,-3/20,1/60],
                  8:[1/280,-4/105,1/5,-4/5,0,4/5,-1/5,4/105,-1/280]}
    elif order==2:
        _coeffs = {2:[1,-2,1],
                   4:[-1/12, 4/3, -5/2, 4/3, -1/12],
                   6:[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90],
                   8:[-1/560, 8/315	, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
            }
    x_axis = x_axis if x_axis is not None else jnp.arange(np.shape(y_axis)[-1])
    return (-1)**order*_derivative(y_axis,x_axis,np.array(_coeffs[accuracy]))

def _derivative(y_axis,x_axis,coeffs):    
    coeffs = np.asarray(coeffs)
    # N        = len(y_axis)
    # pad_width = int(len(coeffs)//2)
    # y_padded = np.pad(y_axis,pad_width,mode='symmetric')
    # x_padded = np.pad(x_axis,pad_width,mode='linear_ramp',
                      # end_values=(-pad_width,N+pad_width-1))
   
    # print(np.shape(y_axis),np.shape(x_axis),np.shape(y_padded),np.shape(h))
    if len(np.shape(y_axis))>1:
        y, x   = jnp.broadcast_arrays(y_axis,x_axis)
        
        
        xcubed   = jnp.power(jnp.diff(x,axis=1),3.0)
        h        = jnp.insert(xcubed,0,xcubed[0][0],axis=1)
        
        L         = np.shape(coeffs)[0]
        coeffs_ = jnp.zeros((L,L))
        coeffs_  = coeffs_.at[L//2].set(coeffs)
        
        y_deriv  = jsp.signal.convolve(y,coeffs_,'same')/h
        # y_deriv  = jsp.signal.convolve2d(coeffs_,y,'same')/h
    else:   
        xcubed   = jnp.power(np.diff(x_axis),3)
        h        = jnp.insert(xcubed,0,xcubed[0])
        y_deriv  = jnp.convolve(y_axis, coeffs, 'same')/h
    
    return y_deriv




def derivative_eval(x,y_array,x_array):
    deriv=derivative(y_array,x_array,order=1,accuracy=8)
    srep =interpolate.splrep(x_array,deriv)
    return interpolate.splev(x,srep) 

def derivative_zero(y_array,x_array,left,right):
    return brentq(derivative_eval,left,right,args=(y_array,x_array))
    # return None
    
    
def error_from_covar(func,pars,covar,x,N=1000):
    samples  = np.random.multivariate_normal(pars,covar,N)
    try:
        values_ = [func(x,*(sample)) for sample in samples]
    except:
        values_ = [func(x,sample) for sample in samples]
    error    = np.std(values_,axis=0)
    return error

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
def integer_slice(i, n, m):
    # return nth to mth digit of i (as int)
    l = mth.floor(mth.log10(i)) + 1
    return i / int(pow(10, l - m)) % int(pow(10, m - n + 1))

def mad(x,around=None):
    ''' Returns median absolute deviation of input array'''
    around = around if around is not None else np.median(x)
    return np.median(np.abs(around-x))
def mean_abs_dev(x):
    ''' Returns the mean absolute deviaton of input array'''
    return np.mean(np.abs(x-np.mean(x)))
def nmoment(x, counts, c, n):
    ''' Calculates the nth moment of x around c using counts as weights'''
    #https://stackoverflow.com/questions/29064053/calculate-moments-mean-variance-of-distribution-in-python
    return np.sum(counts*(x-c)**n) / np.sum(counts)

def polynomial(x, *p):
    y = np.zeros_like(x,dtype=np.float64)
    for i,a in enumerate(p):
        y += a*x**i
    return y
def polyjac(x,*p):
    
#    y = np.zeros((len(p),len(x)),dtype=np.float64)
#    for i,a in enumerate(p):
#        y[i]= i*a*x**(i-1)
    y = np.array([i*a*x**(i-1) for i,a in enumerate(p)])
    return np.atleast_2d(y).T
def rms(x,around_mean=False,axis=None):
    ''' Returns root mean square of input array'''
    mean = np.nanmean(x,axis=axis) if around_mean==True else 0.0
    return np.sqrt(np.nanmean(np.square(x-mean),axis=axis))
def running_mean(x, N,pad_mode='symmetric',convolve_mode='same'):
    
    if convolve_mode=='same':
        x_pad = np.pad(x,N,mode=pad_mode)
        mean = np.convolve(x_pad, np.ones((N,))/N,mode=convolve_mode)
        return mean[N:-N]
    if convolve_mode=='valid':
        return mean
    if convolve_mode=='full':
        mean = np.convolve(x, np.ones((N,))/N,mode=convolve_mode)
        return mean[int(N/2-1):-int(N/2-1)-1]

def running_rms(x, N):
    x2 = np.power(x,2)
    window = np.ones(N)/float(N)
    return np.sqrt(np.convolve(x2, window, 'same'))
def running_std(x, N):
    import pandas as pd
        #return np.convolve(x, np.ones((N,))/N)[(N-1):]
    series = pd.Series(x)
    return series.rolling(N).std()

def round_to_significant(a):
    return round(a, -int(np.floor(np.log10(np.abs(a)))))

def get_significant_digit(a):
    return np.floor(np.log10(np.abs(a)))
def round_to_closest(a,b):
    '''
    a (float, array of floats): number to round
    b (float): closest unit to round to
    '''
    if len(np.shape(a))>0:
        return np.array([round(_/b)*b for _ in a])
    else:
        return round(a/b)*b

def sig_clip(v):
       m1=np.mean(v,axis=-1)
       std1=np.std(v-m1,axis=-1)
       m2=np.mean(v[abs(v-m1)<5*std1],axis=-1)
       std2=np.std(v[abs(v-m2)<5*std1],axis=-1)
       m3=np.mean(v[abs(v-m2)<5*std2],axis=-1)
       std3=np.std(v[abs(v-m3)<5*std2],axis=-1)
       return abs(v-m3)<5*std3 
   
def sigclip1d(v,sigma=3,maxiter=10,converge_num=0.02,plot=False):
    from matplotlib.patches import Polygon
    v    = np.array(v)
    ct   = np.size(v)
    dim  = len(np.shape(v))
    iter = 0; c1 = 1.0; c2=0.0
    while (c1>=c2) and (iter<maxiter):
        lastct = ct
        mean   = np.nanmean(v)
        std    = np.nanstd(v-mean)
        cond   = abs(v-mean)<sigma*std
        cut    = np.where(cond)
        ct     = len(cut[0])
        
        c1     = abs(ct-lastct)
        c2     = converge_num*lastct
        iter  += 1
    if plot:
        if dim == 1:
            plt.figure(figsize=(12,6))
            plt.scatter(np.arange(len(v)),v,s=2,c="C0")        
            plt.scatter(np.arange(len(v))[~cond],v[~cond],
                            s=10,c="C1",marker='x')
            plt.axhline(mean,ls='-',c='r')
            plt.axhline(mean+sigma*std,ls='--',c='r')
            plt.axhline(mean-sigma*std,ls='--',c='r')
        if dim == 2:
            fig, ax = plt.subplots(1)
            im = ax.imshow(v,aspect='auto',
                       vmin=np.percentile(v[cond],3),
                       vmax=np.percentile(v[cond],97),
                       cmap=plt.cm.coolwarm)
            cb = fig.colorbar(im)
            for i,c in enumerate(cond):
                if np.all(c): 
                    continue
                else:
                    print(c)
                    x = [j for j in c if j is False]
                    print(x)
                    ax.add_patch(Polygon([[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]], closed=True,
                      fill=False, hatch='/'))
#            plt.scatter(np.arange(len(v))[~cond],v[~cond],
#                            s=10,c="C1",marker='x')
#            plt.axhline(mean,ls='-',c='r')
#            plt.axhline(mean+sigma*std,ls='--',c='r')
#            plt.axhline(mean-sigma*std,ls='--',c='r')
    return cond


def sigclip1d_biased_low(
    v,
    sigma_lower=3.0,  # Kappa for clipping points below the mean
    sigma_upper=1.5,  # Smaller kappa for clipping points above the mean (more aggressive)
    maxiter=10,
    converge_num=0.02,
    plot=False,
    verbose=False
):
    """
    Performs iterative sigma clipping on a 1D array, biased towards
    keeping lower y-value data points by using a more aggressive
    upper clipping threshold.

    Args:
        v (array-like): Input 1D array of values.
        sigma_lower (float): Number of std devs for the lower clipping bound.
        sigma_upper (float): Number of std devs for the upper clipping bound.
                             A smaller value makes clipping of high points more aggressive.
        maxiter (int): Maximum number of iterations.
        converge_num (float): Convergence criterion (fraction of points).
        plot (bool): Whether to generate a diagnostic plot.
        verbose (bool): If True, print iteration info.

    Returns:
        np.ndarray: A boolean mask of the same shape as the input `v`,
                    where True indicates a point that was kept.
    """
    v_clean = np.asarray(v)[np.isfinite(v)] # Work on a copy, remove NaNs/Infs at start
    if len(v_clean) == 0:
        return np.zeros_like(v, dtype=bool) # Return all False if no valid data

    # Create a mask for the original array `v`
    # Initialize based on finite values, then update from v_clean's clipping
    original_indices = np.arange(len(v))
    finite_mask_orig = np.isfinite(v)
    
    # Map indices from v_clean back to original v
    # This is a bit tricky if NaNs were present.
    # A simpler approach: operate on a boolean mask of the original array 'v'.
    
    keep_mask = np.isfinite(v) # Start with only finite values as potentially kept
    if not np.any(keep_mask): # All NaNs/Infs
        return keep_mask

    v_current_iter = np.asarray(v)[keep_mask] # Data used in current iteration

    ct = np.sum(keep_mask) # Number of currently kept points
    
    for iteration in range(maxiter):
        lastct = ct
        if len(v_current_iter) < 2 : # Need at least 2 points for std
            if verbose: print(f"Iter {iteration+1}: Too few points ({len(v_current_iter)}) to continue clipping.")
            break

        mean_iter = np.mean(v_current_iter)
        std_iter = np.std(v_current_iter)

        if std_iter < 1e-9: # Effectively flat, no more clipping needed or possible
            if verbose: print(f"Iter {iteration+1}: Std dev too small ({std_iter:.2e}). Stopping.")
            break

        # Define asymmetric clipping thresholds
        low_thresh = mean_iter - sigma_lower * std_iter
        high_thresh = mean_iter + sigma_upper * std_iter # More aggressive upper clip

        # Identify points to keep *within the current iteration's data*
        iter_keep_mask = (v_current_iter >= low_thresh) & (v_current_iter <= high_thresh)
        
        # Update the global keep_mask
        # This requires mapping indices from v_current_iter back to the original 'v'
        # This is where it gets a bit complex. A simpler way is to update v_current_iter directly.
        
        v_current_iter = v_current_iter[iter_keep_mask]
        ct = len(v_current_iter)

        if verbose:
            print(f"Iter {iteration+1}: Mean={mean_iter:.2f}, Std={std_iter:.2f}, "
                  f"Bounds=[{low_thresh:.2f}, {high_thresh:.2f}], Kept={ct}/{lastct}")

        # Check for convergence
        if abs(ct - lastct) <= converge_num * lastct:
            if verbose: print(f"Converged after {iteration+1} iterations.")
            break
        if iteration == maxiter - 1 and verbose:
            print("Reached max iterations.")
            
    # Now, construct the final boolean mask for the *original* input array `v`
    # The points in `v_current_iter` are the ones that survived.
    # We need to find which original points correspond to these.
    # This is non-trivial if there are duplicate values in `v`.
    
    # A more robust way is to update the `keep_mask` on the original array in each iteration.
    # Let's re-do the loop with this approach:

    keep_mask_final = np.isfinite(v) # Start with finite values
    if not np.any(keep_mask_final): return keep_mask_final

    for iteration in range(maxiter):
        v_iter_data = v[keep_mask_final] # Get current data subset
        
        if len(v_iter_data) < 2: # Not enough points for std
            if verbose: print(f"Iter {iteration+1} (mask update): Not enough points ({len(v_iter_data)}) to clip.")
            break
            
        num_kept_before_iter = np.sum(keep_mask_final)

        mean_iter = np.mean(v_iter_data)
        std_iter = np.std(v_iter_data)

        if std_iter < 1e-9:
            if verbose: print(f"Iter {iteration+1} (mask update): Std dev too small ({std_iter:.2e}). Stopping.")
            break

        low_thresh = mean_iter - sigma_lower * std_iter
        high_thresh = mean_iter + sigma_upper * std_iter

        # Create a mask for points to *remove* from the current set of kept points
        # Points are removed if they are outside the new bounds *relative to the current mean/std*
        # This needs to be applied to indices of `keep_mask_final` that are True.
        
        indices_currently_kept = np.where(keep_mask_final)[0]
        values_of_kept_points = v[indices_currently_kept]
        
        # Identify which of these `values_of_kept_points` should be *newly clipped*
        newly_clipped_mask_for_kept = (values_of_kept_points < low_thresh) | (values_of_kept_points > high_thresh)
        
        # Get original indices of points to be newly clipped
        indices_to_unkeep = indices_currently_kept[newly_clipped_mask_for_kept]
        
        if verbose:
            print(f"Iter {iteration+1} (mask update): Mean={mean_iter:.2f}, Std={std_iter:.2f}, "
                  f"Bounds=[{low_thresh:.2f}, {high_thresh:.2f}], "
                  f"Points currently kept={num_kept_before_iter}, To unkeep={len(indices_to_unkeep)}")

        if len(indices_to_unkeep) == 0: # No change in this iteration
            if verbose: print(f"Converged after {iteration+1} iterations (mask update).")
            break
            
        keep_mask_final[indices_to_unkeep] = False # Update the main mask
        num_kept_after_iter = np.sum(keep_mask_final)

        if abs(num_kept_after_iter - num_kept_before_iter) <= converge_num * num_kept_before_iter:
            if verbose: print(f"Converged based on converge_num after {iteration+1} iterations (mask update).")
            break
        if iteration == maxiter - 1 and verbose:
            print("Reached max iterations (mask update).")

    # --- Plotting (using the final keep_mask_final) ---
    if plot:
        v_plot = np.asarray(v) # Ensure v is an array for plotting
        dim = v_plot.ndim # Dimension of the original input `v`
        
        # Final mean and std of the kept points for plotting bounds
        final_kept_data = v_plot[keep_mask_final]
        if len(final_kept_data) > 1:
            plot_mean = np.mean(final_kept_data)
            plot_std = np.std(final_kept_data)
            plot_low_thresh = plot_mean - sigma_lower * plot_std
            plot_high_thresh = plot_mean + sigma_upper * plot_std
        else: # Not enough points for mean/std, or all clipped
            plot_mean, plot_std, plot_low_thresh, plot_high_thresh = np.nan, np.nan, np.nan, np.nan

        if dim == 1:
            plt.figure(figsize=(12,6))
            x_coords_orig = np.arange(len(v_plot))
            
            plt.scatter(x_coords_orig, v_plot, s=10, c="grey", alpha=0.5, label="Original Data")
            if np.any(keep_mask_final):
                plt.scatter(x_coords_orig[keep_mask_final], v_plot[keep_mask_final], s=15, c="C0", label="Kept Points")
            if np.any(~keep_mask_final & finite_mask_orig): # Plot points that were finite but clipped
                 plt.scatter(x_coords_orig[~keep_mask_final & finite_mask_orig], 
                             v_plot[~keep_mask_final & finite_mask_orig],
                             s=20, c="C1", marker='x', label="Clipped Points")
            
            if np.isfinite(plot_mean): plt.axhline(plot_mean,ls='-',c='r', label=f"Mean Kept ({plot_mean:.2f})")
            if np.isfinite(plot_low_thresh): plt.axhline(plot_low_thresh,ls='--',c='r', label=f"Lower Clip ({sigma_lower}σ)")
            if np.isfinite(plot_high_thresh): plt.axhline(plot_high_thresh,ls='--',c='r', label=f"Upper Clip ({sigma_upper}σ)")
            plt.title(f"Sigma Clipping (Lower biased: σ_low={sigma_lower}, σ_up={sigma_upper})")
            plt.xlabel("Index"); plt.ylabel("Value"); plt.legend(); plt.grid(True)
            plt.show()
        elif dim == 2: # Original plotting for 2D (needs adaptation if `v` is 2D)
            print("Plotting for 2D input `v` in sigclip1d_biased_low is not fully implemented with this mask logic.")
            # The Polygon part was from your original, and needs x,y indices for the 2D array.
            # This function now primarily assumes 1D input 'v'.
            # If 'v' is 2D, keep_mask_final would also be 2D.
            fig, ax = plt.subplots(1)
            im = ax.imshow(v_plot, aspect='auto', cmap=plt.cm.coolwarm,
                           vmin=np.nanpercentile(final_kept_data,3) if len(final_kept_data)>0 else np.nanmin(v_plot),
                           vmax=np.nanpercentile(final_kept_data,97) if len(final_kept_data)>0 else np.nanmax(v_plot))
            cb = fig.colorbar(im)
            # Example: To show clipped regions (this would need more work for actual polygons)
            # overlay_clipped = np.where(keep_mask_final, np.nan, 1) # Mark clipped areas
            # ax.imshow(overlay_clipped, cmap='Reds', alpha=0.3, aspect='auto')
            ax.set_title(f"2D Sigma Clipping (Lower biased: σ_low={sigma_lower}, σ_up={sigma_upper})")
            plt.show()
            
    return keep_mask_final

def sigclip2d(v,sigma=5,maxiter=100,converge_num=0.02):
    ct   = np.size(v)
    iter = 0; c1 = 1.0; c2=0.0
    while (c1>=c2) and (iter<maxiter):
        lastct = ct
        mean   = np.mean(v)
        std    = np.std(v-mean)
        cond   = abs(v-mean)<sigma*std
        cut    = np.where(cond)
        ct     = len(cut[0])
        
        c1     = abs(ct-lastct)
        c2     = converge_num*lastct
        iter  += 1
    return cond
def negpos(number):
    return -abs(number),abs(number)
def removenan(array):
    return array[~np.isnan(array)]
def nan_to_num(array):
    finite = np.isfinite(array)
    return array[finite]
def round_up_to_odd(f,b=1):
    return round_to_closest(np.ceil(f) // 2 * 2 + 1 ,b)
def round_up_to_even(f,b=1):
    return round_to_closest(np.ceil(f) // 2 * 2 ,b)
def round_down_to_odd(f,b=1):
    return round_to_closest(np.floor(f) // 2 * 2 + 1 ,b)
def round_down_to_even(f,b=1):
    return round_to_closest(np.floor(f) // 2 * 2 ,b)

def missing_elements(L, start, end):
    """
    https://stackoverflow.com/questions/16974047/
    efficient-way-to-find-missing-elements-in-an-integer-sequence
    """
    if end - start <= 1: 
        if L[end] - L[start] > 1:
            yield from range(L[start] + 1, L[end])
        return

    index = start + (end - start) // 2

    # is the lower half consecutive?
    consecutive_low =  L[index] == L[start] + (index - start)
    if not consecutive_low:
        yield from missing_elements(L, start, index)

    # is the upper part consecutive?
    consecutive_high =  L[index] == L[end] - (end - index)
    if not consecutive_high:
        yield from missing_elements(L, index, end)
def find_missing(integers_list,start=None,limit=None):
    """
    Given a list of integers and optionally a start and an end, finds all
    the integers from start to end that are not in the list.

    'start' and 'end' default respectivly to the first and the last item of the list.

    Doctest:

    >>> find_missing([1,2,3,5,6,7], 1, 7)
    [4]

    >>> find_missing([2,3,6,4,8], 2, 8)
    [5, 7]

    >>> find_missing([1,2,3,4], 1, 4)
    []

    >>> find_missing([11,1,1,2,3,2,3,2,3,2,4,5,6,7,8,9],1,11)
    [10]

    >>> find_missing([-1,0,1,3,7,20], -1, 7)
    [2, 4, 5, 6]

    >>> find_missing([-2,0,3], -5, 2)
    [-5, -4, -3, -1, 1, 2]

    >>> find_missing([2],4,5)
    [4, 5]

    >>> find_missing([3,5,6,7,8], -3, 5)
    [-3, -2, -1, 0, 1, 2, 4]

    >>> find_missing([1,2,4])
    [3]

    """
    # https://codereview.stackexchange.com/a/77890
    start = start if start is not None else integers_list[0]
    limit = limit if limit is not None else integers_list[-1]
    return [i for i in range(start,limit + 1) if i not in integers_list]
def average(values,errors=None):
    """
    Return the weighted average and weighted sample standard deviation.

    values, weights -- Numpy ndarrays with the same shape.

    Assumes that weights contains only integers (e.g. how many samples in each group).

    See also https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Frequency_weights
    """
    weights = np.atleast_1d(errors) if errors is not None else np.ones_like(values)
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    variance = variance*sum(weights)/(sum(weights)-1)
    return (average, np.sqrt(variance))
    
# def wmean(values,errors=None):
#     errors = np.atleast_1d(errors) if errors is not None else np.ones_like(values)
#     variance = np.power(errors,2)
#     weights  = 1./variance 
#     mean  = np.nansum(values * weights) / np.nansum(weights)
#     sigma = 1./ np.sqrt(np.sum(weights))
#     return mean,sigma

def wmean(
    values: Union[np.ndarray, Sequence],
    errors: Optional[Union[np.ndarray, Sequence, float]] = None,
    axis: Optional[int] = None,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Calculates the weighted mean of values, optionally along a specified axis,
    and the standard error of that mean.

    Parameters:
    ----------
    values : array_like
        Data to be averaged. NaN values in `values` are ignored (treated as
        having zero weight).
    errors : array_like or float, optional
        Errors (standard deviations) associated with each value in `values`.
        These are used to calculate weights as 1/error^2.
        If None (default), all valid (non-NaN) data points in `values` are
        given equal weight (1.0), effectively performing an unweighted mean.
        Must be broadcastable to the shape of `values` if not None.
        - Zero errors (infinite weights): If one or more data points have
          zero error, these points dominate the mean. The mean will be the
          (unweighted) average of these "perfectly known" points, and
          `sigma_mean` will be 0.0.
        - NaN errors: If an error is NaN, the corresponding data point is
          ignored (treated as having zero weight).
    axis : int, optional
        Axis along which to compute the weighted mean. If None (default),
        computes the mean over the entire flattened array.

    Returns:
    -------
    mean : float or np.ndarray
        The calculated weighted mean. Shape will be `values.shape` with the
        `axis` dimension removed, or scalar if `axis` is None.
    sigma_mean : float or np.ndarray
        The standard error of the weighted mean, calculated as
        1 / sqrt(sum_of_weights).
        - If weights are derived from actual measurement errors (sigma_i),
          this represents the propagated uncertainty in the weighted mean.
        - If `errors` were None (unweighted mean), this becomes
          1 / sqrt(N_valid_points), where N_valid_points is the number of
          non-NaN values contributing to the mean.
        - If "perfect" measurements (zero error) determined the mean,
          `sigma_mean` is 0.0.
        - If sum_of_weights is zero for a slice (e.g., all NaNs or all
          zero-weight points), `mean` will be NaN and `sigma_mean` will be Inf.

    Notes:
    -----
    - The function is designed to handle NaNs gracefully. A NaN in `values`
      or a NaN in `errors` will result in that data point not contributing
      to the mean or sum of weights.
    - The behavior for `errors=None` results in `mean` being equivalent to
      `np.nanmean(values, axis=axis)`. The `sigma_mean` in this case is
      `1 / sqrt(N_valid_points)`.

    Examples:
    --------
    >>> values_arr = np.array([1.0, 2.0, 3.0])
    >>> errors_arr = np.array([0.1, 0.1, 0.2])
    >>> wmean(values_arr, errors_arr)
    (1.4444444444444444, 0.06085806194501844)

    >>> values_nd = np.array([[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]])
    >>> wmean(values_nd, axis=1) # Unweighted, like np.nanmean
    (array([1.5, 5. ]), array([0.70710678, 0.70710678]))

    >>> values_zero_err = np.array([1.0, 2.0, 3.0, 4.0])
    >>> errors_zero_err = np.array([1.0, 0.0, 1.0, 1.0]) # Middle point has zero error
    >>> wmean(values_zero_err, errors_zero_err)
    (2.0, 0.0)

    >>> errors_two_zero_err = np.array([1.0, 0.0, 0.0, 1.0]) # Two points with zero error
    >>> wmean(values_zero_err, errors_two_zero_err) # Mean of 2.0 and 3.0
    (2.5, 0.0)
    
    >>> values_with_nan = np.array([np.nan, 1.0, 2.0])
    >>> errors_all_same = np.array([0.1, 0.1, 0.1])
    >>> wmean(values_with_nan, errors_all_same)
    (1.5, 0.07071067811865475)
    
    >>> wmean([1, 2, np.nan], errors=[0.1, np.inf, 0.1], axis=0) # error=np.inf means weight=0
    (1.0, 0.1)
    """
    # Ensure values are a numpy array of float type for calculations
    _values = np.asarray(values, dtype=float)

    if errors is None:
        # Unweighted mean: initial weights are 1 for all points.
        _weights_intermediate = np.ones_like(_values, dtype=float)
    else:
        _errors = np.asarray(errors, dtype=float)
        if _values.shape != _errors.shape and _errors.size != 1:
            try:
                # Check if errors can be broadcast to values shape
                np.broadcast_shapes(_values.shape, _errors.shape)
            except ValueError as e:
                raise ValueError(
                    f"Shape of values {_values.shape} and errors {_errors.shape} "
                    "are not compatible for broadcasting."
                ) from e
        
        # Calculate variance and then weights (1/variance)
        variance = np.power(_errors, 2)
        with np.errstate(divide='ignore', invalid='ignore'): # Handle division by zero or NaN errors
            _weights_intermediate = 1.0 / variance
    
    # Finalize initial weights:
    # If original value is NaN or initial weight is NaN (e.g., from NaN error),
    # set the weight to NaN. This ensures nansum correctly ignores these points.
    _weights = np.where(
        np.isnan(_values) | np.isnan(_weights_intermediate), 
        np.nan, 
        _weights_intermediate
    )

    # --- Infinite weight handling (for zero errors) ---
    # An infinite weight means the error was zero. These points dominate.
    inf_w_mask = np.isinf(_weights)

    # Determine if any slice (or the whole array if axis=None) contains infinite weights
    if axis is None:
        has_inf_in_slice = np.any(inf_w_mask) # Boolean scalar
    else:
        # has_inf_in_slice will have shape of _weights but with `axis` dimension as 1
        has_inf_in_slice = np.any(inf_w_mask, axis=axis, keepdims=True)

    # Define weights for the "infinite weight" scenario:
    # - Points with original infinite weight get a new weight of 1.0.
    # - Other points get a weight of NaN (to be ignored by nansum).
    # - If original value was NaN (already handled by _weights becoming NaN),
    #   this ensures weights_for_inf_case also becomes NaN there.
    weights_for_inf_case = np.where(inf_w_mask, 1.0, np.nan)
    # Ensure consistency if _values itself was NaN (though inf_w_mask should be False there)
    weights_for_inf_case = np.where(np.isnan(_values), np.nan, weights_for_inf_case)

    # Select the appropriate set of weights for calculation:
    # If a slice has infinite weights, use `weights_for_inf_case` for that slice.
    # Otherwise, use the prepared `_weights`.
    # `has_inf_in_slice` (potentially with keepdims=True) correctly broadcasts for this selection.
    chosen_weights = np.where(has_inf_in_slice, weights_for_inf_case, _weights)

    # --- Calculate mean and sigma_mean ---
    # Numerator for mean: sum of (value * chosen_weight)
    # np.nansum correctly ignores entries where chosen_weights is NaN or value*chosen_weights is NaN
    weighted_sum_val = np.nansum(_values * chosen_weights, axis=axis)
    
    # Denominator for mean: sum of chosen_weights
    sum_of_chosen_weights = np.nansum(chosen_weights, axis=axis)

    # Calculate mean
    with np.errstate(divide='ignore', invalid='ignore'): # Handle sum_of_chosen_weights = 0
        mean_val = weighted_sum_val / sum_of_chosen_weights
    
    # Calculate sigma_mean (standard error of the weighted mean)
    # Default sigma_mean: 1 / sqrt(sum_of_chosen_weights)
    with np.errstate(divide='ignore', invalid='ignore'): # Handle sum_of_chosen_weights = 0
        sigma_mean_default = 1.0 / np.sqrt(sum_of_chosen_weights)

    # If a slice had infinite weights (and thus used weights_for_inf_case),
    # its sigma_mean should be 0.0, indicating a "perfect" determination.
    # `has_inf_in_slice_for_output` needs to match shape of `sigma_mean_default` (output shape)
    if axis is None:
        has_inf_in_slice_for_output = has_inf_in_slice # Already a scalar
    else:
        # Squeeze the summed-over axis from `has_inf_in_slice` (which had keepdims=True)
        has_inf_in_slice_for_output = np.squeeze(has_inf_in_slice, axis=axis)
    
    sigma_mean_val = np.where(has_inf_in_slice_for_output, 0.0, sigma_mean_default)
    
    # If the original input `values` was a scalar (or 0-D array) and axis is None,
    # np.nansum would return Python floats. Otherwise, np.ndarray. This is usually fine.
    # No explicit conversion to scalar item needed unless strict type matching is critical.

    return mean_val, sigma_mean_val


    # return average(values,errors)
def aicc(chisq,n,p):
    ''' Returns the Akiake information criterion value 
    chisq = chi square
    n     = number of points
    p     = number of free parameters
    
    '''
    return chisq + 2*p + 2*p*(p+1)/(n-p-1)

def get_mode(array, axis=0, bins=50,sigma=20, plot=False):
    '''
    Calculates the mode of the array by examining a histogram of the array. 
    Performs supersampling and smoothing of the histogram to increase precision

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.
    bins : TYPE, optional
        DESCRIPTION. The default is 50.
    sigma : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # Ensure input array has at least two dimensions
    if array.ndim < 2:
        return _compute_column_mode(array,bins)

    # Compute mode along each column
    modes = jnp.array([_compute_column_mode(array[:, i], bins,sigma, False) 
                       for i in range(array.shape[1])])

    return modes



def _compute_column_mode(column, bins, sigma=20, plot=False):
    
    # Compute histogram
    hist, edges = jnp.histogram(column, bins=bins)

    # Compute midpoints of bins
    mids = 0.5 * (edges[1:] + edges[:-1])

    # Step 1: Supersample histogram array
    supersampled_hist = jnp.repeat(hist, 20)
    supersampled_mids = jnp.linspace(edges.min(), edges.max(), len(supersampled_hist))

    # Step 2: Smooth histogram array with Gaussian kernel
    smoothed_hist = gaussian_filter1d(supersampled_hist, sigma=sigma)

    # Step 3: Calculate first derivative of the smoothed array
    deriv_smoothed_hist = jnp.gradient(smoothed_hist)

    # Step 4: Find index corresponding to the mode of the smoothed array
    mode_index = jnp.argmax(smoothed_hist)
    

    # Step 5: Find the location of the mode using the derivative of the smoothed array
    mode = _find_mode_location(deriv_smoothed_hist, supersampled_mids, mode_index)

    # Step 6: Return the mode location
    # mode = supersampled_mids[mode_location]
    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.stairs(hist, edges)
        plt.plot(supersampled_mids, supersampled_hist)
        plt.plot(supersampled_mids, smoothed_hist)
        # plt.plot(supersampled_mids, deriv_smoothed_hist*100)
        plt.axvline(mode)
        plt.show()
    # if mode>1.: mode=1.
    return mode

def _find_mode_location(deriv_smoothed_hist, supersampled_mids, mode_index,
                        window=5):
    # Find the crossing point by linear interpolation
    if mode_index == 0:
        left_index = 0
    else:
        left_index = mode_index-window
    if mode_index==len(deriv_smoothed_hist)-1:
        right_index = len(deriv_smoothed_hist)-1
    else:
        right_index = mode_index + (window-1)
    
    y_0 = deriv_smoothed_hist[left_index]
    y_1 = deriv_smoothed_hist[right_index]
    x_0 = supersampled_mids[left_index]
    x_1 = supersampled_mids[right_index]
    
    
    # Linearly interpolate to find the crossing point
    slope = (y_1 - y_0) / (x_1 - x_0)
    crossing_point = x_0 - y_0 / slope
    
    if crossing_point<jnp.min(supersampled_mids):
        crossing_point = supersampled_mids[0]
    if crossing_point>jnp.max(supersampled_mids):
        crossing_point = supersampled_mids[-1]
    return crossing_point