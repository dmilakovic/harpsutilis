#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 18:06:41 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from harps.functions import math as mathfunc
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.cluster import DBSCAN
from collections import Counter
from matplotlib.lines import Line2D
from sklearn.linear_model import RANSACRegressor, HuberRegressor
import scipy
from scipy.signal import welch, windows, convolve, wiener, nuttall # windows, convolve used in placeholders


__all__ = [
        "get_window",
        "get_window_robust",
        "detect_initial_extrema_candidates",
        "filter_extrema_with_dbscan",
        "apply_clustering_filter_to_extrema",
        "plot_smoothing_and_derivatives",
        "plot_clustering_2d_results",
        "plot_feature_space_3d",
        "process_spectrum_for_lfc_lines",
        ]


# ==============================================================================
# == SECTION 1: USER-PROVIDED HELPER FUNCTIONS (YOU NEED TO FILL THESE IN) ====
# ==============================================================================


def _datacheck(x_axis, y_axis):
    if x_axis is None:
        x_axis = np.arange(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError( 
                "Input vectors y_axis and x_axis must have same length")
    
    # cut = np.where(y_axis!=0.0)[0]
    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    

def _rebin( a, newshape ):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape), ValueError(f"Array ndim {a.ndim} does not match newshape len {len(newshape)}")

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = np.mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

def _smooth(x, window_len=11, window="hanning", mode="valid"):
    """
    smooth the data using a window of the requested size.
    
    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    
    keyword arguments:
    x -- the input signal 
    
    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)
    
    window -- the type of window from 'flat', 'hanning', 'hamming', 
        'bartlett', 'blackman', where flat is a moving average
        (default: 'hanning')

    
    return: the smoothed signal
        
    example:

    t = linspace(-2,2,0.1)
    x = sin(t)+randn(len(t))*0.1
    y = _smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, 
    numpy.convolve, scipy.signal.lfilter 
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len<3:
        return x
    #declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
        "nuttall": nuttall
        }
    
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w = window_funcs[window](window_len)
    except KeyError:
        raise ValueError(
            "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
            *window_funcs.keys()))
    
    y = np.convolve(w / w.sum(), s, mode = mode)
    
    if mode=='valid':
        trim = (window_len - 1) // 2
        return y [trim : -trim]
    elif mode=='same':
        pad = window_len - 1
        return y[pad : pad + len(x)]

def get_window(y_axis,plot=False,minimum=8):
    if len(y_axis) == 0: return minimum
    
    freq0, P0    = welch(y_axis,nperseg=min(len(y_axis), 512))
    cut = np.where(freq0>0.02)[0]
    if len(cut) == 0 and len(freq0) > 0 : # If cut removes everything, take the first non-DC if available
        if freq0[0] == 0 and len(freq0)>1: cut = [1]
        elif freq0[0] > 0: cut = [0]
        else: return minimum # No usable frequencies
    elif len(cut) == 0 and len(freq0) == 0: # No frequencies at all
         return minimum
     
    freq, P = freq0[cut], P0[cut]
    maxind     = np.argmax(P)
    maxfreq    = freq[maxind]
    if plot:
        import matplotlib.ticker as ticker
        def one_over(x):
            """Vectorized 1/x, treating x==0 manually"""
            x = np.array(x, float)
            near_zero = np.isclose(x, 0)
            x[near_zero] = np.inf
            x[~near_zero] = 1 / x[~near_zero]
            return x
        
        fig, ax = plt.subplots(1)
        ax.semilogy(1/freq,P)
        ax.semilogy(1/maxfreq,P[maxind],marker='x',c='C1')
        ax2 = ax.secondary_xaxis('top',functions=(one_over,one_over))
        # ax2.xaxis.set_major_locator(ticker.LogLocator(base=10,numticks=10))
        ax.set_xlabel("Period (pix)")
        ax2.set_xlabel("Frequency (1/pix)")
        ax.set_ylabel("Power")
        # ax.set_xlim(1e-3,0.51)
        
        
    if maxfreq <= 1e-9: return minimum
    window = mathfunc.round_down_to_odd(1. / maxfreq)
    return window if window > minimum else minimum


# ==============================================================================
# == SECTION 2: ROBUST HELPER FUNCTIONS (Likely from our previous discussions) =
# ==============================================================================

def get_window_robust(y_axis, sampling_rate=1.0, plot=False,
                      default_window_period=15,
                      min_period_pixels=3, max_period_pixels=None,
                      nperseg_min_periods=4):
    if not isinstance(y_axis, np.ndarray): y_axis = np.asarray(y_axis)
    if len(y_axis) < 2 * default_window_period:
        return mathfunc.round_down_to_odd(max(3, len(y_axis) // 5))

    if max_period_pixels is None: max_period_pixels = len(y_axis) / 3.0
    max_period_pixels = max(max_period_pixels, min_period_pixels * 2.0)

    nperseg_val = min(len(y_axis), max(256, int(nperseg_min_periods * max_period_pixels)))
    if nperseg_val <=0: nperseg_val = min(len(y_axis), 256) # Safety for nperseg

    try:
        freqs, Pxx = welch(y_axis, fs=sampling_rate, nperseg=nperseg_val, scaling='density', window='hann')
    except ValueError:
        return mathfunc.round_down_to_odd(default_window_period)

    min_freq_of_interest = 1.0 / (max_period_pixels + 1e-9)
    max_freq_of_interest = 1.0 / (min_period_pixels + 1e-9)
    
    # Exclude DC (freqs[0] can be zero) and frequencies outside our band
    # Ensure freqs are positive before taking reciprocal for period
    valid_indices = np.where((freqs > min_freq_of_interest) & 
                             (freqs < max_freq_of_interest) & 
                             (freqs > 1e-9))[0]

    dominant_freq = 0
    if len(valid_indices) == 0 or len(Pxx[valid_indices]) == 0:
        period = default_window_period
    else:
        peak_idx_in_valid = np.argmax(Pxx[valid_indices])
        dominant_freq = freqs[valid_indices[peak_idx_in_valid]]
        if dominant_freq <= 1e-9: period = default_window_period
        else: period = 1.0 / dominant_freq

    period = np.clip(period, min_period_pixels, max_period_pixels)
    window_len = mathfunc.round_down_to_odd(int(period))
    window_len = max(3, window_len)

    if plot:
        fig, ax = plt.subplots(1, figsize=(8,5))
        ax.semilogy(freqs[freqs>1e-9], Pxx[freqs>1e-9], label='Full Spectrum (Welch)', color='grey', alpha=0.7)
        if len(valid_indices) > 0:
             ax.semilogy(freqs[valid_indices], Pxx[valid_indices], label='Considered Band', color='orange')
             if dominant_freq > 1e-9:
                ax.semilogy(dominant_freq, Pxx[valid_indices[peak_idx_in_valid]], 'rx', markersize=10,
                             label=f'Dominant (Period ~{period:.1f}px)')
        ax.set_xlabel(f'Frequency (1/pixels, fs={sampling_rate})')
        ax.set_ylabel('PSD'); ax.legend()
        ax.set_title(f'Power Spectrum for Window Estimation (Est. Window: {window_len})')
        ax.grid(True, linestyle=':'); plt.tight_layout(); plt.show()
    return window_len

def get_window_robust_targeted(
    y_axis,
    sampling_rate=1.0,
    plot=False,
    # Target period range for the desired window length
    target_min_period_pixels=10,
    target_max_period_pixels=20,
    # Fallback if no suitable peak in target range
    default_window_period=15,
    # Broader constraints for Welch and initial period estimation
    overall_min_period_pixels=3, # Smallest possible period to consider in PSD
    overall_max_period_pixels=None, # Largest, e.g., for nperseg estimation
    nperseg_min_periods=4,
    verbose=False
):
    if not isinstance(y_axis, np.ndarray):
        y_axis = np.asarray(y_axis)

    if len(y_axis) == 0: # Handle empty input
        print("Warning: y_axis is empty. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)
        
    if len(y_axis) < 2 * target_max_period_pixels : # Heuristic if signal is too short for target
        print(f"Warning: y_axis (len {len(y_axis)}) too short for target period {target_max_period_pixels}. "
              f"Using a fraction of y_axis length or default.")
        # Fallback to a smaller window if signal is very short
        # Or simply use default_window_period if that's preferred
        return mathfunc.round_down_to_odd(max(overall_min_period_pixels, min(target_min_period_pixels, len(y_axis) // 5)))


    # Determine overall_max_period_pixels for nperseg calculation if not provided
    if overall_max_period_pixels is None:
        overall_max_period_pixels = len(y_axis) / 3.0
    # Ensure overall_max is at least as large as target_max
    overall_max_period_pixels = max(overall_max_period_pixels, target_max_period_pixels, overall_min_period_pixels * 2.0)


    nperseg_val = min(len(y_axis), max(256, int(nperseg_min_periods * overall_max_period_pixels)))
    if nperseg_val <= 0:
        nperseg_val = min(len(y_axis), 256) # Safety for nperseg
    if nperseg_val > len(y_axis): # nperseg cannot be greater than signal length
        nperseg_val = len(y_axis)


    try:
        freqs, Pxx = welch(y_axis, fs=sampling_rate, nperseg=nperseg_val, scaling='density', window='hann')
    except ValueError as e:
        print(f"ValueError during Welch: {e}. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)

    if freqs.size == 0: # Welch returned no frequencies
        print("Warning: Welch returned no frequencies. Returning default_window_period.")
        return mathfunc.round_down_to_odd(default_window_period)

    # Define the target frequency range based on the target period range
    # Add epsilon to avoid division by zero or issues with exact boundaries
    target_max_freq = 1.0 / (target_min_period_pixels - 1e-9) if target_min_period_pixels > 1e-9 else np.inf
    target_min_freq = 1.0 / (target_max_period_pixels + 1e-9) if target_max_period_pixels > 1e-9 else 0

    # Find indices of frequencies within our *target* band
    # Also ensure freqs > 0 to avoid issues with 1/freq
    target_band_indices = np.where(
        (freqs >= target_min_freq) & (freqs <= target_max_freq) & (freqs > 1e-9)
    )[0]

    period_in_target_range_found = False
    selected_period = default_window_period # Initialize with fallback
    dominant_freq_in_target_band = 0

    if target_band_indices.size > 0 and Pxx[target_band_indices].size > 0:
        # Find the peak power within this specific target frequency band
        peak_idx_in_target_band = np.argmax(Pxx[target_band_indices])
        dominant_freq_in_target_band = freqs[target_band_indices[peak_idx_in_target_band]]

        if dominant_freq_in_target_band > 1e-9:
            selected_period = 1.0 / dominant_freq_in_target_band
            period_in_target_range_found = True
            if verbose: print(f"Found dominant peak in target range: Period = {selected_period:.2f} pixels (Freq = {dominant_freq_in_target_band:.3f})")
        else:
            print("Dominant frequency in target band is too low or zero.")
    else:
        print(f"No significant power or no frequencies found in the target period range ({target_min_period_pixels}-{target_max_period_pixels} pixels).")

    if not period_in_target_range_found:
        print(f"Falling back to default_window_period: {default_window_period} pixels.")
        selected_period = default_window_period
        # Optional: could also fall back to overall max power if default is not preferred,
        # but that might defeat the purpose of the target range.
        # For example, to use the previous logic as a fallback:
        # overall_min_freq = 1.0 / (overall_max_period_pixels + 1e-9)
        # overall_max_freq = 1.0 / (overall_min_period_pixels + 1e-9)
        # overall_valid_indices = np.where((freqs > overall_min_freq) & (freqs < overall_max_freq) & (freqs > 1e-9))[0]
        # if overall_valid_indices.size > 0 and Pxx[overall_valid_indices].size > 0:
        #     overall_peak_idx = np.argmax(Pxx[overall_valid_indices])
        #     overall_dominant_freq = freqs[overall_valid_indices[overall_peak_idx]]
        #     if overall_dominant_freq > 1e-9: selected_period = 1.0 / overall_dominant_freq


    # Final window length must still be an odd integer and within overall sensible bounds
    # Clip the selected_period to be within the target range itself, or if not found,
    # it will be the default_window_period (which should ideally be within or near target range).
    # The clipping below to overall_min/max_period_pixels is more of a safety for the final window_len.
    final_selected_period = np.clip(selected_period, target_min_period_pixels, target_max_period_pixels)
    # However, if the peak was outside, and you MUST use the target range, the above clip is correct.
    # If selected_period was a fallback (like default_window_period), this clip ensures it's within target.

    # If the goal was "if there's a peak in target, use it, otherwise use global max (clipped to overall)",
    # then the final_selected_period logic would be:
    # if period_in_target_range_found:
    #    final_selected_period = selected_period # Already from target band
    # else:
    #    # Recalculate global max period based on wider overall_min/max_period_pixels
    #    # This part is essentially your original function's logic
    #    overall_min_f = 1.0 / (overall_max_period_pixels + 1e-9)
    #    overall_max_f = 1.0 / (overall_min_period_pixels + 1e-9)
    #    overall_indices = np.where((freqs > overall_min_f) & (freqs < overall_max_f) & (freqs > 1e-9))[0]
    #    if overall_indices.size > 0:
    #        glob_peak_idx = np.argmax(Pxx[overall_indices])
    #        glob_dom_freq = freqs[overall_indices[glob_peak_idx]]
    #        final_selected_period = 1.0 / glob_dom_freq if glob_dom_freq > 1e-9 else default_window_period
    #    else:
    #        final_selected_period = default_window_period
    # final_selected_period = np.clip(final_selected_period, overall_min_period_pixels, overall_max_period_pixels)
    # The above commented block is if you want global peak as fallback.
    # The current simpler logic (using default_window_period as fallback and clipping to target) is:
    final_selected_period = np.clip(selected_period, target_min_period_pixels, target_max_period_pixels)


    window_len = mathfunc.round_down_to_odd(int(final_selected_period))
    window_len = max(3, window_len) # Ensure at least 3

    if plot:
        fig, ax = plt.subplots(1, figsize=(10, 6)) # Increased figure size for more details
        ax.semilogy(freqs[freqs > 1e-9], Pxx[freqs > 1e-9], label='Full Spectrum (Welch)', color='grey', alpha=0.5, zorder=1)

        # Highlight the target frequency band
        ax.axvspan(target_min_freq, target_max_freq, alpha=0.2, color='lightgreen', label=f'Target Freq. Band ({target_min_period_pixels}-{target_max_period_pixels} px period)')

        if target_band_indices.size > 0:
            ax.semilogy(freqs[target_band_indices], Pxx[target_band_indices], label='Power in Target Band', color='orange', zorder=2,linewidth=1.5)
            if dominant_freq_in_target_band > 1e-9 and period_in_target_range_found:
                ax.semilogy(dominant_freq_in_target_band, Pxx[target_band_indices[peak_idx_in_target_band]],
                            'rx', markersize=12, mew=2,
                            label=f'Peak in Target (Period ~{1.0/dominant_freq_in_target_band:.1f}px)', zorder=3)
        
        # Optionally, plot the global peak if it was different
        # This requires calculating it if not period_in_target_range_found
        # For simplicity, this part is omitted but could be added for diagnostics

        ax.set_xlabel(f'Frequency (1/pixels, fs={sampling_rate})')
        ax.set_ylabel('PSD'); ax.legend(loc='best') # Changed legend location
        ax.set_title(f'Window Estimation (Targeted: {target_min_period_pixels}-{target_max_period_pixels} px). Final Est. Window: {window_len} px', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

        # Secondary x-axis for period
        def freq_to_period(f_arr):
            # Ensure f_arr is an array for vectorized operations
            f_arr = np.asarray(f_arr)
            # Handle potential zero or negative frequencies to avoid warnings/errors
            periods = np.full_like(f_arr, np.inf, dtype=float)
            valid_f = f_arr > 1e-9
            periods[valid_f] = 1.0 / f_arr[valid_f]
            return periods

        # secax = ax.secondary_xaxis('top', functions=(freq_to_period, freq_to_period))
        # secax.set_xlabel('Period (pixels)')

        # This section tries to set specific ticks, but the problem is often with the auto-derived limits before this.
        common_periods = np.array([overall_min_period_pixels, target_min_period_pixels, (target_min_period_pixels+target_max_period_pixels)/2 , target_max_period_pixels, overall_max_period_pixels])
        common_periods = np.unique(common_periods[common_periods > 0])
        # If common_periods is empty or results in inf after transformation by the axis itself, it can be an issue.
        # However, the error "Axis limits cannot be NaN or Inf" points to the automatic limit setting.
        # secax.set_xticks(common_periods)
        # secax.set_xticklabels([f"{p:.0f}" for p in common_periods])

        # plt.tight_layout(); 
        plt.show()
    return window_len

def _ensure_alternation(maxima, minima):
    peaks_x_np, peaks_y_np     = _datacheck(*maxima)
    valleys_x_np, valleys_y_np = _datacheck(*minima)

    if not peaks_x_np.size and not valleys_x_np.size: return np.array([]), np.array([]), np.array([]), np.array([])
    if not peaks_x_np.size: return np.array([]), np.array([]), valleys_x_np, valleys_y_np
    if not valleys_x_np.size: return peaks_x_np, peaks_y_np, np.array([]), np.array([])

    all_x = np.concatenate((peaks_x_np, valleys_x_np)); all_y = np.concatenate((peaks_y_np, valleys_y_np))
    all_types = np.concatenate((np.ones(peaks_x_np.size), -np.ones(valleys_x_np.size)))
    
    sorted_indices = np.argsort(all_x)
    sorted_x, sorted_y, sorted_types = all_x[sorted_indices], all_y[sorted_indices], all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list = [], [], []
    if not sorted_x.size: return np.array([]), np.array([]), np.array([]), np.array([])

    final_x_list.append(sorted_x[0]); 
    final_y_list.append(sorted_y[0]); 
    final_types_list.append(sorted_types[0])
    for i in range(1, sorted_x.size):
        current_x, current_y, current_type = sorted_x[i], sorted_y[i], sorted_types[i]
        last_type = final_types_list[-1]
        if current_type != last_type:
            final_x_list.append(current_x); 
            final_y_list.append(current_y); 
            final_types_list.append(current_type)
        else:
            if current_type == 1 and current_y > final_y_list[-1]: 
                final_x_list[-1], final_y_list[-1] = current_x, current_y
            elif current_type == -1 and current_y < final_y_list[-1]: 
                final_x_list[-1], final_y_list[-1] = current_x, current_y
    
    final_x_arr = np.array(final_x_list); 
    final_y_arr = np.array(final_y_list); 
    final_types_arr = np.array(final_types_list)
    
    out_peaks_x = final_x_arr[final_types_arr == 1]; 
    out_peaks_y = final_y_arr[final_types_arr == 1]
    out_valleys_x = final_x_arr[final_types_arr == -1]; 
    out_valleys_y = final_y_arr[final_types_arr == -1]
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y

def _calculate_features_for_extrema(extrema_x, extrema_y, opposite_x, opposite_y, is_peak,
                                    min_extrema_for_spacing_model=3):
    extrema_x, extrema_y = np.asarray(extrema_x), np.asarray(extrema_y)
    opposite_x, opposite_y = np.asarray(opposite_x), np.asarray(opposite_y)
    num_extrema = extrema_x.size
    if num_extrema == 0: return np.array([]).reshape(0, 3)

    prom_depth_values = np.zeros(num_extrema)
    for i in range(num_extrema):
        x_i, y_i = extrema_x[i], extrema_y[i]
        left_indices = np.where(opposite_x < x_i)[0]; right_indices = np.where(opposite_x > x_i)[0]
        y_brackets_list = []
        if left_indices.size > 0: y_brackets_list.append(opposite_y[left_indices[-1]])
        if right_indices.size > 0: y_brackets_list.append(opposite_y[right_indices[0]])
        if not y_brackets_list: prom_depth_values[i] = 0; continue
        y_brackets_arr = np.array(y_brackets_list)
        prom_depth_values[i] = (y_i - np.max(y_brackets_arr)) if is_peak else (np.min(y_brackets_arr) - y_i)
        prom_depth_values[i] = max(0, prom_depth_values[i])

    spacing_dev_values = np.zeros(num_extrema)
    if num_extrema >= min_extrema_for_spacing_model and num_extrema >= 2: # Need at least 2 points for diff
        spacings = np.diff(extrema_x)
        if spacings.size >= 1 : # Need at least 1 spacing
             spacing_x_midpoints = (extrema_x[:-1] + extrema_x[1:]) / 2.0
             if spacing_x_midpoints.size >= 2: # TheilSen needs at least 2 points
                model = TheilSenRegressor(random_state=42)
                try:
                    model.fit(spacing_x_midpoints.reshape(-1, 1), spacings)
                    for i in range(num_extrema):
                        devs_list = []
                        if i > 0:
                            d_prev = extrema_x[i] - extrema_x[i-1]; x_mid_prev = (extrema_x[i] + extrema_x[i-1]) / 2.0
                            d_prev_exp = model.predict(np.array([[x_mid_prev]]))[0]; devs_list.append(abs(d_prev - d_prev_exp))
                        if i < num_extrema - 1:
                            d_next = extrema_x[i+1] - extrema_x[i]; x_mid_next = (extrema_x[i+1] + extrema_x[i]) / 2.0
                            d_next_exp = model.predict(np.array([[x_mid_next]]))[0]; devs_list.append(abs(d_next - d_next_exp))
                        if devs_list: spacing_dev_values[i] = np.mean(devs_list)
                except (ValueError, np.linalg.LinAlgError): pass # Model fit failed
    feature_matrix = np.vstack([extrema_x, prom_depth_values, spacing_dev_values]).T
    return feature_matrix

# ==============================================================================
# == SECTION 3: REFACTORED CORE LOGIC FUNCTIONS ================================
# ==============================================================================

def detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed,
                                      deriv_method='coeff',
                                      first_deriv_zero_threshold_factor=0.05):
    if len(y_smoothed) == 0: # Guard against empty input
        return np.array([]), np.array([]), np.array([]), np.array([])

    derivative1st = mathfunc.derivative1d(y_smoothed, x=None, order=1, method=deriv_method)
    derivative2nd = mathfunc.derivative1d(y_smoothed, x=None, order=2, method=deriv_method)

    if len(derivative1st) == 0: # Guard against empty derivative (e.g. if y_smoothed was too short)
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(derivative1st)) if derivative1st.size > 0 else 1e-9
    candidate_indices = []
    
    # Ensure derivative1st has at least 2 elements for np.diff
    if derivative1st.size >= 2:
        sign_changes = np.diff(np.sign(derivative1st))
        crossings_idx = np.where(np.abs(sign_changes) == 2)[0] # Indices *before* change
        for idx in crossings_idx:
            # Ensure idx+1 is within bounds for derivative1st
            if idx + 1 < derivative1st.size:
                candidate_indices.append(idx if np.abs(derivative1st[idx]) < np.abs(derivative1st[idx + 1]) else idx + 1)
            else: # Only idx is valid
                candidate_indices.append(idx)

    candidate_indices = sorted(list(set(candidate_indices)))
    candidate_indices = np.array(candidate_indices, dtype=int)

    if candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Ensure candidate indices are valid for derivative2nd and x_coords_for_smoothed
    valid_mask = (candidate_indices < derivative2nd.size) & \
                 (candidate_indices < x_coords_for_smoothed.size)
    valid_candidate_indices = candidate_indices[valid_mask]
    if valid_candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] < 0]
    min_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] > 0]

    peaks_x = x_coords_for_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    peaks_y = y_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    valleys_x = x_coords_for_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    valleys_y = y_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    
    return peaks_x, peaks_y, valleys_x, valleys_y

def filter_extrema_with_dbscan(extrema_x, extrema_y, scaled_features,
                               dbscan_eps, dbscan_min_samples):
    extrema_x, extrema_y = np.asarray(extrema_x), np.asarray(extrema_y)
    if scaled_features is None or scaled_features.shape[0] == 0:
        return extrema_x, extrema_y, None

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels = dbscan.fit_predict(scaled_features)

    core_point_labels = labels[labels != -1]
    if core_point_labels.size > 0:
        most_common_label = Counter(core_point_labels).most_common(1)[0][0]
        true_indices = np.where(labels == most_common_label)[0]
        return extrema_x[true_indices], extrema_y[true_indices], labels
    else:
        return np.array([]), np.array([]), labels

def apply_clustering_filter_to_extrema(
    cand_peaks_x, cand_peaks_y, cand_valleys_x, cand_valleys_y,
    dbscan_eps_peaks, dbscan_min_samples_peaks,
    dbscan_eps_valleys, dbscan_min_samples_valleys,
    min_features_for_clustering, min_extrema_for_spacing_model
):
    # Store original candidates that go into this specific filtering stage for plotting
    plot_feed_peaks_x, plot_feed_peaks_y = np.copy(cand_peaks_x), np.copy(cand_peaks_y)
    plot_feed_valleys_x, plot_feed_valleys_y = np.copy(cand_valleys_x), np.copy(cand_valleys_y)

    # --- Filter Peaks ---
    final_peaks_x, final_peaks_y = cand_peaks_x, cand_peaks_y # Default if no clustering
    scaled_peak_features, peak_labels = None, None
    if cand_peaks_x.size >= min_features_for_clustering and cand_valleys_x.size > 0:
        peak_features_unscaled = _calculate_features_for_extrema(
            cand_peaks_x, cand_peaks_y, cand_valleys_x, cand_valleys_y,
            is_peak=True, min_extrema_for_spacing_model=min_extrema_for_spacing_model
        )
        if peak_features_unscaled.shape[0] > 0:
            scaler_p = RobustScaler()
            scaled_peak_features = scaler_p.fit_transform(peak_features_unscaled)
            final_peaks_x, final_peaks_y, peak_labels = filter_extrema_with_dbscan(
                cand_peaks_x, cand_peaks_y, scaled_peak_features,
                dbscan_eps_peaks, dbscan_min_samples_peaks
            )
    plot_data_peaks = (scaled_peak_features, peak_labels, plot_feed_peaks_x, plot_feed_peaks_y)

    # --- Alternation before Valley Filtering ---
    peaks_after_peak_filt_x, peaks_after_peak_filt_y, \
    valleys_for_valley_filt_x, valleys_for_valley_filt_y = _ensure_alternation(
        (final_peaks_x, final_peaks_y), (cand_valleys_x, cand_valleys_y)
    )
    plot_feed_valleys_x, plot_feed_valleys_y = np.copy(valleys_for_valley_filt_x), np.copy(valleys_for_valley_filt_y)


    # --- Filter Valleys ---
    final_valleys_x, final_valleys_y = valleys_for_valley_filt_x, valleys_for_valley_filt_y
    scaled_valley_features, valley_labels = None, None
    if valleys_for_valley_filt_x.size >= min_features_for_clustering and peaks_after_peak_filt_x.size > 0:
        valley_features_unscaled = _calculate_features_for_extrema(
            valleys_for_valley_filt_x, valleys_for_valley_filt_y,
            peaks_after_peak_filt_x, peaks_after_peak_filt_y,
            is_peak=False, min_extrema_for_spacing_model=min_extrema_for_spacing_model
        )
        if valley_features_unscaled.shape[0] > 0:
            scaler_v = RobustScaler()
            scaled_valley_features = scaler_v.fit_transform(valley_features_unscaled)
            final_valleys_x, final_valleys_y, valley_labels = filter_extrema_with_dbscan(
                valleys_for_valley_filt_x, valleys_for_valley_filt_y, scaled_valley_features,
                dbscan_eps_valleys, dbscan_min_samples_valleys
            )
    plot_data_valleys = (scaled_valley_features, valley_labels, plot_feed_valleys_x, plot_feed_valleys_y)

    # --- Final Alternation ---
    final_peaks_x_alt, final_peaks_y_alt, \
    final_valleys_x_alt, final_valleys_y_alt = _ensure_alternation(
        (final_peaks_x, final_peaks_y), (final_valleys_x, final_valleys_y)
    )
    
    return (final_peaks_x_alt, final_peaks_y_alt, final_valleys_x_alt, final_valleys_y_alt,
            plot_data_peaks, plot_data_valleys)


def robust_local_noise_std(y_data, center_idx, window_half_width, fallback_noise_std):
    """Estimates local noise std around center_idx, excluding a central region."""
    if len(y_data) == 0: return fallback_noise_std
    
    # Define regions to the left and right of the peak/valley for noise estimation
    # Exclude the very central part where the peak/valley itself is
    exclude_half_width = window_half_width // 3 # Exclude a smaller region around the extremum
    
    left_start = max(0, center_idx - window_half_width)
    left_end = max(0, center_idx - exclude_half_width)
    
    right_start = min(len(y_data), center_idx + exclude_half_width + 1)
    right_end = min(len(y_data), center_idx + window_half_width + 1)
    
    noise_samples = []
    if left_end > left_start:
        noise_samples.extend(y_data[left_start:left_end])
    if right_end > right_start:
        noise_samples.extend(y_data[right_start:right_end])
        
    if len(noise_samples) < 5: # Not enough samples for robust MAD
        return fallback_noise_std
        
    mad_val = scipy.stats.median_abs_deviation(noise_samples, scale='normal')
    return mad_val if mad_val > 1e-9 else fallback_noise_std

def robust_noise_std(y):
    if len(y)==0: return 1e-6
    mad = scipy.stats.median_abs_deviation(y, scale='normal'); return mad if mad > 1e-9 else 1e-6
    
def _fit_spacing_polynomial(extrema_x_coords, poly_degree=2):
    """
    Fits a polynomial to predict expected spacing S_exp(x_midpoint).
    Returns the polynomial model object.
    """
    extrema_x_coords = np.asarray(extrema_x_coords)
    if len(extrema_x_coords) < poly_degree + 2 : # Need enough points to fit spacings
        return None # Cannot fit model

    spacings = np.diff(extrema_x_coords)
    x_midpoints = (extrema_x_coords[:-1] + extrema_x_coords[1:]) / 2.0

    if len(x_midpoints) < poly_degree + 1: # Need enough (midpoint, spacing) pairs
        return None

    # Prepare data for polynomial regression
    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
    X_poly = poly_features.fit_transform(x_midpoints.reshape(-1, 1))
    
    # Use a robust regressor
    # model = HuberRegressor(epsilon=1.35) # Less sensitive to outliers than OLS
    model = RANSACRegressor(random_state=42) # Good for significant outliers
    try:
        model.fit(X_poly, spacings)
        return model, poly_features # Return both model and feature transformer
    except ValueError: # Can happen if data is degenerate
        return None, None


def _calculate_features_for_extrema_v3( # Version 3: S/N and new spacing dev
    all_extrema_x, all_extrema_y, all_extrema_types, # Types: +1 for peak, -1 for valley
    # For Prominence/Depth S/N
    original_y_data_for_noise, original_x_data_for_noise,
    noise_estimation_window_pixels, global_fallback_noise_std,
    # For Spacing Deviation
    spacing_poly_degree=2, spacing_uncertainty_const=2.0
):
    num_total_extrema = len(all_extrema_x)
    if num_total_extrema == 0: return np.array([]).reshape(0, 3)

    feature_x_coords = np.asarray(all_extrema_x)
    feature_prom_depth_snr = np.zeros(num_total_extrema)
    feature_spacing_dev_norm = np.zeros(num_total_extrema)

    # --- Calculate Prominence/Depth S/N for all extrema ---
    # This requires bracketing, so we need to process peaks and valleys somewhat separately here
    # or have a way to find "opposite type" neighbors for each point in the combined list.
    # For simplicity, let's assume we first get raw prom/depth, then S/N.
    
    raw_prom_depth_values = np.zeros(num_total_extrema)
    # Need to iterate and find bracketing for each point based on its type
    # This is complex if they are mixed. It's easier if we pass separated peak/valley lists
    # to a sub-function that calculates this specific feature.
    #
    # Let's defer detailed S/N calculation to when we call this from apply_clustering...
    # and assume for now it's pre-calculated or we use a placeholder.
    # For a combined list, we'd need to find, for each peak, its bracketing valleys,
    # and for each valley, its bracketing peaks from the combined list.
    # This means sorting by x and then iterating.
    
    # Placeholder for Prominence/Depth S/N - this part needs careful implementation
    # if truly processing a mixed list here.
    # The original _calculate_features_for_extrema_v2 did this by taking separated lists.
    # For now, let's just use a dummy value or assume it's passed in.
    # We'll actually calculate this properly in apply_clustering_filter_to_extrema_v3 before calling this.
    # So, this function will receive pre-calculated prom_depth_snr.
    #
    # THEREFORE, THIS FUNCTION'S SIGNATURE AND ROLE CHANGES SLIGHTLY:
    # It will take pre-calculated prom_depth_snr and calculate spacing_dev.

    # Let's rename this function to reflect its new role more accurately:
    # _calculate_spacing_deviation_feature(extrema_x_sorted, spacing_poly_degree, spacing_uncertainty_const)
    # And the S/N calculation will be separate.

    # For now, keeping it in one function and making a simplification for S/N:
    # Assume all_extrema_types helps us.
    for i in range(num_total_extrema):
        x_i, y_i, type_i = all_extrema_x[i], all_extrema_y[i], all_extrema_types[i]
        # Simplified: find closest opposite types (not strictly bracketing for mixed list)
        opposite_type_mask = (all_extrema_types == -type_i)
        if np.any(opposite_type_mask):
            opposite_x_candidates = all_extrema_x[opposite_type_mask]
            opposite_y_candidates = all_extrema_y[opposite_type_mask]
            
            # Find opposites to left and right
            left_opp_idx = np.where(opposite_x_candidates < x_i)[0]
            right_opp_idx = np.where(opposite_x_candidates > x_i)[0]
            
            y_brackets = []
            if left_opp_idx.size > 0: y_brackets.append(opposite_y_candidates[left_opp_idx[-1]])
            if right_opp_idx.size > 0: y_brackets.append(opposite_y_candidates[right_opp_idx[0]])

            if y_brackets:
                raw_pd = (y_i - np.max(y_brackets)) if type_i == 1 else (np.min(y_brackets) - y_i)
                raw_prom_depth_values[i] = max(0, raw_pd)
        
        # S/N calculation
        center_idx_in_orig = np.argmin(np.abs(original_x_data_for_noise - x_i)) if original_x_data_for_noise is not None else 0
        local_noise = robust_local_noise_std(original_y_data_for_noise, center_idx_in_orig, noise_estimation_window_pixels // 2, global_fallback_noise_std) if original_y_data_for_noise is not None else global_fallback_noise_std
        feature_prom_depth_snr[i] = raw_prom_depth_values[i] / local_noise if local_noise > 1e-9 else raw_prom_depth_values[i] / 1e-9


    # --- Calculate Spacing Deviation (Normalized by Uncertainty) ---
    # Fit polynomial to ALL spacings first (peaks and valleys together or separately?)
    # If clustering together, a single spacing model from all extrema_x might be best.
    spacing_model, spacing_poly_transformer = _fit_spacing_polynomial(all_extrema_x, poly_degree=spacing_poly_degree)

    if spacing_model is not None:
        for i in range(num_total_extrema):
            x_i = all_extrema_x[i]
            devs_norm_list = []
            # Backward spacing
            if i > 0:
                s_bwd_actual = x_i - all_extrema_x[i-1]
                x_mid_bwd = (x_i + all_extrema_x[i-1]) / 2.0
                x_mid_bwd_poly = spacing_poly_transformer.transform(np.array([[x_mid_bwd]]))
                s_bwd_expected = spacing_model.predict(x_mid_bwd_poly)[0]
                devs_norm_list.append((s_bwd_actual - s_bwd_expected) / spacing_uncertainty_const)
            # Forward spacing
            if i < num_total_extrema - 1:
                s_fwd_actual = all_extrema_x[i+1] - x_i
                x_mid_fwd = (x_i + all_extrema_x[i+1]) / 2.0
                x_mid_fwd_poly = spacing_poly_transformer.transform(np.array([[x_mid_fwd]]))
                s_fwd_expected = spacing_model.predict(x_mid_fwd_poly)[0]
                devs_norm_list.append((s_fwd_actual - s_fwd_expected) / spacing_uncertainty_const)
            
            if devs_norm_list:
                feature_spacing_dev_norm[i] = np.mean(np.abs(devs_norm_list)) # Use mean absolute normalized deviation
            # Else, it remains 0 (e.g., for isolated points)
    else:
        print("    Warning: Could not fit spacing model. Spacing deviation feature will be zero.")

    feature_matrix = np.vstack([
        feature_x_coords, 
        feature_prom_depth_snr,
        feature_spacing_dev_norm
    ]).T
    return feature_matrix

def _ensure_alternation_and_tag(peaks_x, peaks_y, valleys_x, valleys_y):
    """
    Ensures strict P-V-P-V alternation AND returns a combined list with type tags.
    Type tags: +1 for peak, -1 for valley.
    """
    peaks_x_np, peaks_y_np = np.asarray(peaks_x), np.asarray(peaks_y)
    valleys_x_np, valleys_y_np = np.asarray(valleys_x), np.asarray(valleys_y)

    if not peaks_x_np.size and not valleys_x_np.size:
        return np.array([]), np.array([]), np.array([]), \
               np.array([]), np.array([]), np.array([]) # Empty x,y,type for combined

    # Combine all features with a type indicator
    all_x_coords = []
    all_y_coords = []
    all_types_list = []

    if peaks_x_np.size > 0:
        all_x_coords.extend(peaks_x_np)
        all_y_coords.extend(peaks_y_np)
        all_types_list.extend([1] * peaks_x_np.size)
    if valleys_x_np.size > 0:
        all_x_coords.extend(valleys_x_np)
        all_y_coords.extend(valleys_y_np)
        all_types_list.extend([-1] * valleys_x_np.size)
    
    all_x = np.array(all_x_coords)
    all_y = np.array(all_y_coords)
    all_types = np.array(all_types_list)

    if not all_x.size: # Should be caught by earlier check, but defensive
        return np.array([]), np.array([]), np.array([]), \
               np.array([]), np.array([]), np.array([])

    sorted_indices = np.argsort(all_x)
    sorted_x = all_x[sorted_indices]
    sorted_y = all_y[sorted_indices]
    sorted_types = all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list_out = [], [], []

    if sorted_x.size > 0:
        final_x_list.append(sorted_x[0])
        final_y_list.append(sorted_y[0])
        final_types_list_out.append(sorted_types[0])

        for i in range(1, sorted_x.size):
            current_x, current_y, current_type = sorted_x[i], sorted_y[i], sorted_types[i]
            last_x, last_y, last_type = final_x_list[-1], final_y_list[-1], final_types_list_out[-1]

            if current_type != last_type:
                final_x_list.append(current_x)
                final_y_list.append(current_y)
                final_types_list_out.append(current_type)
            else: # Same type as previous
                if current_type == 1: # Both are peaks
                    if current_y > last_y: # Current peak is higher
                        final_x_list[-1], final_y_list[-1] = current_x, current_y
                else: # Both are valleys
                    if current_y < last_y: # Current valley is lower
                        final_x_list[-1], final_y_list[-1] = current_x, current_y
    
    combined_x_alt = np.array(final_x_list)
    combined_y_alt = np.array(final_y_list)
    combined_types_alt = np.array(final_types_list_out)
    
    # Separate back for return if needed, but also return combined for direct use
    out_peaks_x = combined_x_alt[combined_types_alt == 1]
    out_peaks_y = combined_y_alt[combined_types_alt == 1]
    out_valleys_x = combined_x_alt[combined_types_alt == -1]
    out_valleys_y = combined_y_alt[combined_types_alt == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y, \
           combined_x_alt, combined_y_alt, combined_types_alt


def apply_clustering_filter_to_extrema_v3(
    initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
    dbscan_eps, dbscan_min_samples, min_features_for_clustering,
    original_y_data_for_noise, original_x_data_for_noise, # These are now for the *full* spectrum/segment
    global_noise_std_for_fallback, noise_estimation_window_pixels_feat,
    spacing_poly_degree_feat, spacing_uncertainty_const_feat,
    scaling_options: dict = None
):
    # ... (Full implementation from previous response, calling _calc_feat_v3) ...
    _,_,_,_, all_cand_x,all_cand_y,all_cand_types = _ensure_alternation_and_tag(initial_peaks_x,initial_peaks_y,initial_valleys_x,initial_valleys_y)
    if all_cand_x.size < min_features_for_clustering:
        print(f"    Not enough combined cands ({all_cand_x.size}) for clust. Ret all alt cands.")
        fp_x=all_cand_x[all_cand_types==1];fp_y=all_cand_y[all_cand_types==1];fv_x=all_cand_x[all_cand_types==-1];fv_y=all_cand_y[all_cand_types==-1]
        return (fp_x,fp_y,fv_x,fv_y, (None,None,all_cand_x,all_cand_y,all_cand_types)) # Plot data now tuple

    features_unscaled = _calculate_features_for_extrema_v3(all_cand_x,all_cand_y,all_cand_types,original_y_data_for_noise,original_x_data_for_noise,noise_estimation_window_pixels_feat,global_noise_std_for_fallback,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
    scaled_features_combined=None; labels=None; cl_x,cl_y,cl_t=all_cand_x,all_cand_y,all_cand_types # Default to all if no clustering
    if features_unscaled.shape[0]>0 and features_unscaled.shape[1]==3:
        if scaling_options is None: scaling_options={'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
        def get_s(sname):
            if sname=='minmax':return MinMaxScaler()
            if sname=='standard':return StandardScaler()
            if sname=='robust':return RobustScaler()
            raise ValueError(f"Unk scaler: {sname}")
        scols=[]
        if 'x' in scaling_options and features_unscaled[:,0:1].size > 0: scols.append(get_s(scaling_options['x']).fit_transform(features_unscaled[:,0:1]))
        if 'prom_snr' in scaling_options and features_unscaled[:,1:2].size > 0: scols.append(get_s(scaling_options['prom_snr']).fit_transform(features_unscaled[:,1:2]))
        if 'spacing_dev' in scaling_options and features_unscaled[:,2:3].size > 0: scols.append(get_s(scaling_options['spacing_dev']).fit_transform(features_unscaled[:,2:3]))
        
        if not scols or len(scols) != features_unscaled.shape[1] :
             print("Warning: Feature scaling issue, using RobustScaler on all or features empty.")
             if features_unscaled.size > 0: scaled_features_combined=RobustScaler().fit_transform(features_unscaled)
             else: scaled_features_combined = features_unscaled # Pass empty through
        else: scaled_features_combined=np.hstack(scols)

        if scaled_features_combined.shape[0] > 0 : # Only run DBSCAN if there are scaled features
            dbscan=DBSCAN(eps=dbscan_eps,min_samples=dbscan_min_samples); labels=dbscan.fit_predict(scaled_features_combined)
            core_lbls=labels[labels!=-1]
            if core_lbls.size>0:
                mc_lbl=Counter(core_lbls).most_common(1)[0][0]; true_idx=np.where(labels==mc_lbl)[0]
                cl_x=all_cand_x[true_idx];cl_y=all_cand_y[true_idx];cl_t=all_cand_types[true_idx]
            else: cl_x,cl_y,cl_t=np.array([]),np.array([]),np.array([]) # All outliers
        else: # No scaled features, effectively means no valid input for DBSCAN
             cl_x,cl_y,cl_t=np.array([]),np.array([]),np.array([])


    fp_x=cl_x[cl_t==1];fp_y=cl_y[cl_t==1];fv_x=cl_x[cl_t==-1];fv_y=cl_y[cl_t==-1]
    fpa_x,fpa_y,fva_x,fva_y = _ensure_alternation((fp_x,fp_y),(fv_x,fv_y))
    pdata_comb=(features_unscaled if 'features_unscaled' in locals() and features_unscaled is not None else None,
                scaled_features_combined if scaled_features_combined is not None else None,
                labels if labels is not None else None,
                np.copy(all_cand_x),np.copy(all_cand_y),np.copy(all_cand_types))
    return (fpa_x,fpa_y,fva_x,fva_y,pdata_comb)


# ==============================================================================
# == SECTION 4: PLOTTING UTILITIES =============================================
# ==============================================================================

def plot_smoothing_and_derivatives(
    x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
    derivative1st, derivative2nd, # Pass these in if calculated in orchestrator
    initial_derivative_peaks_x, initial_derivative_peaks_y,
    initial_derivative_valleys_x, initial_derivative_valleys_y,
    actual_smoothing_window_len, deriv1_abs_thresh # Pass absolute threshold
):
    fig_sd, (ax_sd1, ax_sd2, ax_sd3) = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
    fig_sd.suptitle("Spectrum Smoothing and Derivatives", fontsize=14)
    ax_sd1.plot(x_axis_orig, y_axis_orig, label='Original Data', color='lightgray', alpha=0.7)
    ax_sd1.plot(x_coords_for_smoothed, y_smoothed, label=f'Smoothed (Win={actual_smoothing_window_len})', color='blue')
    ax_sd1.scatter(initial_derivative_peaks_x, initial_derivative_peaks_y, marker='o', edgecolor='magenta', facecolor='none', s=40, label='Initial Peak Deriv. Cand.', zorder=3, alpha=0.7)
    ax_sd1.scatter(initial_derivative_valleys_x, initial_derivative_valleys_y, marker='o', edgecolor='cyan', facecolor='none', s=40, label='Initial Valley Deriv. Cand.', zorder=3, alpha=0.7)
    ax_sd1.legend(loc='upper right'); ax_sd1.set_ylabel("Flux")

    ax_sd2.plot(x_coords_for_smoothed, derivative1st, label='1st Derivative')
    ax_sd2.axhline(0, color='gray', linestyle='--'); ax_sd2.axhline(deriv1_abs_thresh, color='orange', linestyle=':', label=f'D1 Thresh'); ax_sd2.axhline(-deriv1_abs_thresh, color='orange', linestyle=':')
    ax_sd2.legend(loc='upper right'); ax_sd2.set_ylabel("1st Deriv")

    ax_sd3.plot(x_coords_for_smoothed, derivative2nd, label='2nd Derivative')
    ax_sd3.axhline(0, color='gray', linestyle='--')
    ax_sd3.legend(loc='upper right'); ax_sd3.set_ylabel("2nd Deriv")
    ax_sd3.set_xlabel("X-coordinate")
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def plot_clustering_2d_results(
        x_axis_orig, y_axis_orig,
        x_coords_for_plot_bg, y_values_for_plot_bg, # e.g., x_smoothed, y_smoothed
        cand_peaks_x, cand_peaks_y, cand_valleys_x, cand_valleys_y, # Candidates fed to clustering
        final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y
        ):
    plt.figure(figsize=(15, 7))
    plt.title("Clustering Filter Results on Spectrum", fontsize=14)
    plt.plot(x_axis_orig, y_axis_orig, label='Original data', 
             drawstyle='steps-mid',
             color='gray', alpha=0.8, zorder=1)
    plt.plot(x_coords_for_plot_bg, y_values_for_plot_bg, label='Smoothed data', 
             drawstyle='steps-mid',
             color='gray', alpha=0.5, zorder=1)

    # Ensure candidates are numpy arrays for .size attribute
    cand_peaks_x, cand_peaks_y = np.asarray(cand_peaks_x), np.asarray(cand_peaks_y)
    cand_valleys_x, cand_valleys_y = np.asarray(cand_valleys_x), np.asarray(cand_valleys_y)
    final_peaks_x, final_peaks_y = np.asarray(final_peaks_x), np.asarray(final_peaks_y)
    final_valleys_x, final_valleys_y = np.asarray(final_valleys_x), np.asarray(final_valleys_y)


    plt.scatter(cand_peaks_x, cand_peaks_y, color='pink', marker='o', s=50, label='Peak Cand. (to Clust.)', zorder=2, alpha=0.6)
    plt.scatter(cand_valleys_x, cand_valleys_y, color='lightblue', marker='o', s=50, label='Valley Cand. (to Clust.)', zorder=2, alpha=0.6)

    set_final_peaks = set(zip(np.round(final_peaks_x, decimals=5), np.round(final_peaks_y, decimals=5)))
    set_final_valleys = set(zip(np.round(final_valleys_x, decimals=5), np.round(final_valleys_y, decimals=5)))

    # --- Corrected section for false peaks ---
    false_peaks_list = []
    if cand_peaks_x.size > 0:
        false_peaks_list = [(x, y) for x, y in zip(cand_peaks_x, cand_peaks_y) if (round(x, 5), round(y, 5)) not in set_final_peaks]

    if false_peaks_list: # Check if the list is not empty
        false_clust_peaks_x, false_clust_peaks_y = zip(*false_peaks_list)
    else:
        false_clust_peaks_x, false_clust_peaks_y = [], []
    # --- End of corrected section for false peaks ---

    # --- Corrected section for false valleys ---
    false_valleys_list = []
    if cand_valleys_x.size > 0:
        false_valleys_list = [(x, y) for x, y in zip(cand_valleys_x, cand_valleys_y) if (round(x, 5), round(y, 5)) not in set_final_valleys]

    if false_valleys_list: # Check if the list is not empty
        false_clust_valleys_x, false_clust_valleys_y = zip(*false_valleys_list)
    else:
        false_clust_valleys_x, false_clust_valleys_y = [], []
    # --- End of corrected section for false valleys ---
    
    plt.scatter(false_clust_peaks_x, false_clust_peaks_y, color='red', marker='x', s=100, label='Discarded Peaks (Clust.)', zorder=3)
    plt.scatter(false_clust_valleys_x, false_clust_valleys_y, color='magenta', marker='x', s=100, label='Discarded Valleys (Clust.)', zorder=3)

    if final_peaks_x.size > 0: plt.scatter(final_peaks_x, final_peaks_y, edgecolor='blue', facecolor='none', marker='o', s=120, label='Final Peaks', zorder=4, linewidth=1.5)
    if final_valleys_x.size > 0: plt.scatter(final_valleys_x, final_valleys_y, edgecolor='green', facecolor='none', marker='o', s=120, label='Final Valleys', zorder=4, linewidth=1.5)
    
    plt.xlabel('X-coordinate'); plt.ylabel('Flux Value (on Context Data Scale)'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.7); plt.show()

def plot_feature_space_3d(scaled_features, labels, is_peak):
    if scaled_features is None or labels is None or scaled_features.shape[0] == 0:
        print(f"No data for 3D feature plot ({'Peaks' if is_peak else 'Valleys'}).")
        return

    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    title_suffix = "Peaks" if is_peak else "Valleys"
    color_kept = 'blue' if is_peak else 'green'
    color_discarded = 'red' if is_peak else 'magenta'

    plot_colors = np.array([color_discarded] * len(labels), dtype='<U7') # Default to discarded
    if len(labels[labels != -1]) > 0:
        mc_label = Counter(labels[labels != -1]).most_common(1)[0][0]
        plot_colors[labels == mc_label] = color_kept
    
    ax_3d.scatter(scaled_features[:, 0], scaled_features[:, 1], scaled_features[:, 2], c=plot_colors, marker='o')
    ax_3d.set_xlabel('Scaled X'); ax_3d.set_ylabel(f'Scaled {"Prominence" if is_peak else "Depth"}'); ax_3d.set_zlabel('Scaled Spacing Dev')
    ax_3d.set_title(f'3D Feature Space for {title_suffix} (Clustering)')
    
    kept_proxy = plt.Line2D([0],[0], linestyle="none", c=color_kept, marker='o')
    discarded_proxy = plt.Line2D([0],[0], linestyle="none", c=color_discarded, marker='o')
    ax_3d.legend([kept_proxy, discarded_proxy], ['Kept (Main Cluster)', 'Discarded'], numpoints=1)
    plt.show()
    
def plot_feature_space_3d_v2( # Renamed to v2 to indicate update
    scaled_features, # Combined scaled features for peaks and valleys
    labels,          # DBSCAN labels for these combined features
    original_types,  # Array indicating type (+1 for peak, -1 for valley) for each feature point
    title_suffix=""  # Optional suffix for the plot title (e.g., segment info)
):
    """
    Plots the 3D feature space for combined peaks and valleys,
    coloring points based on DBSCAN cluster membership and original type.

    Args:
        scaled_features (np.ndarray): Shape (N, 3) - [scaled_x, scaled_prom_depth_snr, scaled_spacing_dev].
        labels (np.ndarray): Shape (N,) - DBSCAN cluster labels.
        original_types (np.ndarray): Shape (N,) - +1 for peak, -1 for valley.
        title_suffix (str): Optional string to append to the plot title.
    """
    if scaled_features is None or labels is None or original_types is None or \
       scaled_features.shape[0] == 0 or scaled_features.shape[0] != len(labels) or \
       scaled_features.shape[0] != len(original_types):
        print(f"No valid data or mismatched array lengths for 3D feature plot. Suffix: {title_suffix}")
        if scaled_features is not None: print(f"  Scaled features shape: {scaled_features.shape}")
        if labels is not None: print(f"  Labels length: {len(labels)}")
        if original_types is not None: print(f"  Original types length: {len(original_types)}")
        return

    fig_3d = plt.figure(figsize=(12, 9)) # Slightly larger for better viewing
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Determine colors based on cluster label (kept/discarded) and marker based on type (peak/valley)
    # Kept: points belonging to the largest non-outlier cluster
    # Discarded: outliers (label -1) or points in other smaller clusters

    kept_mask = np.zeros_like(labels, dtype=bool)
    if len(labels[labels != -1]) > 0:
        # Find the label of the largest non-outlier cluster
        core_point_labels = labels[labels != -1]
        if core_point_labels.size > 0: # Check if there are any core points
            most_common_label = Counter(core_point_labels).most_common(1)[0][0]
            kept_mask = (labels == most_common_label)
    
    # Scatter plot for kept peaks
    kept_peaks_mask = kept_mask & (original_types == 1)
    if np.any(kept_peaks_mask):
        ax_3d.scatter(scaled_features[kept_peaks_mask, 0], 
                      scaled_features[kept_peaks_mask, 1], 
                      scaled_features[kept_peaks_mask, 2], 
                      c='blue', marker='^', s=50, alpha=0.7, label='Kept Peaks (Main Cluster)')

    # Scatter plot for kept valleys
    kept_valleys_mask = kept_mask & (original_types == -1)
    if np.any(kept_valleys_mask):
        ax_3d.scatter(scaled_features[kept_valleys_mask, 0], 
                      scaled_features[kept_valleys_mask, 1], 
                      scaled_features[kept_valleys_mask, 2], 
                      c='green', marker='v', s=50, alpha=0.7, label='Kept Valleys (Main Cluster)')

    # Scatter plot for discarded peaks
    discarded_peaks_mask = (~kept_mask) & (original_types == 1)
    if np.any(discarded_peaks_mask):
        ax_3d.scatter(scaled_features[discarded_peaks_mask, 0], 
                      scaled_features[discarded_peaks_mask, 1], 
                      scaled_features[discarded_peaks_mask, 2], 
                      c='red', marker='^', s=30, alpha=0.4, label='Discarded Peaks')

    # Scatter plot for discarded valleys
    discarded_valleys_mask = (~kept_mask) & (original_types == -1)
    if np.any(discarded_valleys_mask):
        ax_3d.scatter(scaled_features[discarded_valleys_mask, 0], 
                      scaled_features[discarded_valleys_mask, 1], 
                      scaled_features[discarded_valleys_mask, 2], 
                      c='magenta', marker='v', s=30, alpha=0.4, label='Discarded Valleys')

    ax_3d.set_xlabel('Scaled X-coordinate')
    ax_3d.set_ylabel('Scaled Prominence/Depth (S/N)')
    ax_3d.set_zlabel('Scaled Spacing Deviation')
    
    main_title = f'3D Feature Space (Combined Clustering)'
    if title_suffix:
        main_title += f' - {title_suffix}'
    ax_3d.set_title(main_title)
    
    # Create legend handles manually if any points were plotted
    handles = []
    if np.any(kept_peaks_mask): handles.append(Line2D([0],[0], linestyle="none", c='blue', marker='^', label='Kept Peaks'))
    if np.any(kept_valleys_mask): handles.append(Line2D([0],[0], linestyle="none", c='green', marker='v', label='Kept Valleys'))
    if np.any(discarded_peaks_mask): handles.append(Line2D([0],[0], linestyle="none", c='red', marker='^', label='Discarded Peaks', alpha=0.6))
    if np.any(discarded_valleys_mask): handles.append(Line2D([0],[0], linestyle="none", c='magenta', marker='v', label='Discarded Valleys', alpha=0.6))
    
    if handles:
        ax_3d.legend(handles=handles, loc='best')
    else:
        ax_3d.text2D(0.5, 0.5, "No data points to plot in 3D features.", 
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax_3d.transAxes)


    plt.show()
    
#=====
# == SECTION 4B: 
# ========

def refine_extrema_to_original_data(
    detected_peaks_x, detected_peaks_y,  # From process_spectrum_for_lfc_lines (on smoothed/rebinned scale)
    detected_valleys_x, detected_valleys_y,
    original_x_axis, original_y_axis,
    search_window_orig_pixels=3 # Number of original pixels around mapped x to search for true local extremum
):
    """
    Refines the x and y coordinates of detected extrema to the original data grid
    and original flux values.

    The x-coordinates are mapped to the nearest original x-coordinate.
    The y-coordinates are then taken from the original_y_axis at that mapped x,
    potentially refined by looking for the true local extremum in original_y_axis
    within a small window around the mapped x.

    Args:
        detected_peaks_x (list or np.array): X-coords of peaks from smoothed data.
        detected_peaks_y (list or np.array): Y-coords of peaks from smoothed data (can be ignored).
        detected_valleys_x (list or np.array): X-coords of valleys from smoothed data.
        detected_valleys_y (list or np.array): Y-coords of valleys from smoothed data (can be ignored).
        original_x_axis (np.array): The original x-coordinates of the spectrum.
        original_y_axis (np.array): The original y-flux values of the spectrum.
        search_window_orig_pixels (int): After mapping an extremum's x to the original grid,
                                         search within +/- this many original pixels in
                                         original_y_axis to find the actual local max/min.
                                         Set to 0 or 1 to just take the value at the nearest mapped x.

    Returns:
        tuple: (refined_peaks_x, refined_peaks_y, refined_valleys_x, refined_valleys_y)
               where coordinates are on the original data scale and y-values are from original_y_axis.
               Lists are sorted by x-coordinate.
    """
    original_x_axis = np.asarray(original_x_axis)
    original_y_axis = np.asarray(original_y_axis)

    refined_peaks = []
    refined_valleys = []

    if not original_x_axis.size or not original_y_axis.size:
        print("Warning: Original x or y axis is empty. Cannot refine extrema.")
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    # --- Refine Peaks ---
    if detected_peaks_x is not None and len(detected_peaks_x) > 0:
        detected_peaks_x_arr = np.asarray(detected_peaks_x)
        # For each detected peak, find the index of the closest x in original_x_axis
        # np.searchsorted assumes original_x_axis is sorted.
        # It finds insertion points; we need to check neighbors for closest.
        insertion_indices = np.searchsorted(original_x_axis, detected_peaks_x_arr, side='left')

        for i, det_peak_x in enumerate(detected_peaks_x_arr):
            # Determine the actual closest index in original_x_axis
            # Handles cases where insertion_idx is 0 or len(original_x_axis)
            idx = insertion_indices[i]
            if idx == 0:
                closest_orig_idx = 0
            elif idx == len(original_x_axis):
                closest_orig_idx = len(original_x_axis) - 1
            else:
                # Compare distance to original_x_axis[idx-1] and original_x_axis[idx]
                if abs(original_x_axis[idx-1] - det_peak_x) < abs(original_x_axis[idx] - det_peak_x):
                    closest_orig_idx = idx - 1
                else:
                    closest_orig_idx = idx
            
            # Now, search in a small window around closest_orig_idx in original_y_axis
            # for the true local maximum.
            if search_window_orig_pixels > 0:
                win_start = max(0, closest_orig_idx - search_window_orig_pixels)
                win_end = min(len(original_y_axis), closest_orig_idx + search_window_orig_pixels + 1)
                
                if win_start >= win_end: # Should not happen if original_y_axis is not empty
                    local_max_idx_in_window = 0 # fallback
                else:
                    local_max_idx_in_window = np.argmax(original_y_axis[win_start:win_end])
                
                refined_idx = win_start + local_max_idx_in_window
            else: # Just take the y-value at the directly mapped x-coordinate
                refined_idx = closest_orig_idx
            
            refined_peaks.append((original_x_axis[refined_idx], original_y_axis[refined_idx]))

    # --- Refine Valleys ---
    if detected_valleys_x is not None and len(detected_valleys_x) > 0:
        detected_valleys_x_arr = np.asarray(detected_valleys_x)
        insertion_indices = np.searchsorted(original_x_axis, detected_valleys_x_arr, side='left')

        for i, det_valley_x in enumerate(detected_valleys_x_arr):
            idx = insertion_indices[i]
            if idx == 0:
                closest_orig_idx = 0
            elif idx == len(original_x_axis):
                closest_orig_idx = len(original_x_axis) - 1
            else:
                if abs(original_x_axis[idx-1] - det_valley_x) < abs(original_x_axis[idx] - det_valley_x):
                    closest_orig_idx = idx - 1
                else:
                    closest_orig_idx = idx
            
            if search_window_orig_pixels > 0:
                win_start = max(0, closest_orig_idx - search_window_orig_pixels)
                win_end = min(len(original_y_axis), closest_orig_idx + search_window_orig_pixels + 1)

                if win_start >= win_end:
                    local_min_idx_in_window = 0 # fallback
                else:
                    local_min_idx_in_window = np.argmin(original_y_axis[win_start:win_end])

                refined_idx = win_start + local_min_idx_in_window
            else:
                refined_idx = closest_orig_idx

            refined_valleys.append((original_x_axis[refined_idx], original_y_axis[refined_idx]))

    # Remove duplicate extrema that might arise from mapping multiple smoothed extrema
    # to the same original data point, keeping the one that's more "extreme"
    # (though the search window logic should largely handle this)
    # This also ensures sorted output.

    unique_refined_peaks = []
    if refined_peaks:
        # Sort by x, then by y descending (for peaks, keep highest if x is same)
        refined_peaks.sort(key=lambda p: (p[0], -p[1]))
        unique_refined_peaks.append(refined_peaks[0])
        for i in range(1, len(refined_peaks)):
            # If x-coordinate is different from the last added unique peak
            if not np.isclose(refined_peaks[i][0], unique_refined_peaks[-1][0]):
                unique_refined_peaks.append(refined_peaks[i])
            # If x is same, the sort order already picked the more prominent one.

    unique_refined_valleys = []
    if refined_valleys:
        # Sort by x, then by y ascending (for valleys, keep lowest if x is same)
        refined_valleys.sort(key=lambda p: (p[0], p[1]))
        unique_refined_valleys.append(refined_valleys[0])
        for i in range(1, len(refined_valleys)):
            if not np.isclose(refined_valleys[i][0], unique_refined_valleys[-1][0]):
                unique_refined_valleys.append(refined_valleys[i])

    # Unzip to separate x and y lists/arrays
    if unique_refined_peaks:
        r_peaks_x, r_peaks_y = zip(*unique_refined_peaks)
    else:
        r_peaks_x, r_peaks_y = [], []

    if unique_refined_valleys:
        r_valleys_x, r_valleys_y = zip(*unique_refined_valleys)
    else:
        r_valleys_x, r_valleys_y = [], []
        
    return (np.array(r_peaks_x), np.array(r_peaks_y),
            np.array(r_valleys_x), np.array(r_valleys_y))

# ==============================================================================
# == SECTION 5: MAIN ORCHESTRATING FUNCTION ====================================
# ==============================================================================

def process_spectrum_for_lfc_lines(
    y_axis, x_axis=None,
    super_sample_factor=5,
    window_len_method='auto_robust', user_window_len_orig_scale=None,
    gw_min_period_pixels=5, gw_max_period_pixels=None, gw_default_window_period=15,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.05,
    dbscan_eps_peaks=0.5, dbscan_min_samples_peaks=5,
    dbscan_eps_valleys=0.5, dbscan_min_samples_valleys=5,
    min_features_for_clustering=10, min_extrema_for_spacing_model=3,
    plot=False, plot_spectrum_and_derivs_flag=True, plot_clustering_details_flag=True,
    verbose=False
):
    # --- 1. Initial Setup & Smoothing ---
    x_axis_orig, y_axis_orig = _datacheck(x_axis, y_axis)
    if super_sample_factor > 1:
        y_rebinned = _rebin(y_axis_orig, newshape=(int(super_sample_factor * len(y_axis_orig)),)) # Ensure int for shape
        x_rebinned = np.linspace(np.min(x_axis_orig), np.max(x_axis_orig), len(y_rebinned))
    else:
        y_rebinned = np.copy(y_axis_orig); x_rebinned = np.copy(x_axis_orig)

    plot_get_window = plot and plot_spectrum_and_derivs_flag # Pass flag to get_window_robust
    if window_len_method == 'auto_robust':
        window_len_on_orig_scale = get_window_robust_targeted(
            y_axis_orig,
            sampling_rate=1.0,
            plot=plot_get_window,
            # Target period range for the desired window length
            target_min_period_pixels=10,
            target_max_period_pixels=20,
            # Fallback if no suitable peak in target range
            default_window_period=gw_default_window_period,
            # Broader constraints for Welch and initial period estimation
            overall_min_period_pixels=gw_min_period_pixels, # Smallest possible period to consider in PSD
            overall_max_period_pixels=gw_max_period_pixels, # Largest, e.g., for nperseg estimation
        )
    elif window_len_method == 'auto_original_get_window':
        window_len_on_orig_scale = get_window(y_axis_orig, plot=plot_get_window) # Your original
    elif window_len_method == 'user_defined' and user_window_len_orig_scale is not None:
        window_len_on_orig_scale = mathfunc.round_down_to_odd(user_window_len_orig_scale)
    else:
        print("Warning: Invalid window_len_method or missing user_window_len_orig_scale. Using default.")
        window_len_on_orig_scale = get_window_robust_targeted(
            y_axis_orig,
            sampling_rate=1.0,
            plot=plot_get_window,
            # Target period range for the desired window length
            target_min_period_pixels=10,
            target_max_period_pixels=20,
            # Fallback if no suitable peak in target range
            default_window_period=gw_default_window_period,
            # Broader constraints for Welch and initial period estimation
            overall_min_period_pixels=gw_min_period_pixels, # Smallest possible period to consider in PSD
            overall_max_period_pixels=gw_max_period_pixels, # Largest, e.g., for nperseg estimation
        )

    actual_smoothing_window_len = mathfunc.round_down_to_odd(
        int(window_len_on_orig_scale * super_sample_factor)
    )
    # Cap window length
    if len(y_rebinned) > 0: # Ensure y_rebinned is not empty
        actual_smoothing_window_len = min(actual_smoothing_window_len, mathfunc.round_down_to_odd(len(y_rebinned) // 2 -1 if len(y_rebinned) // 2 -1 >=1 else 1))
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose:
        print(f"Orig scale window: {window_len_on_orig_scale}, Rebinned smooth window: {actual_smoothing_window_len}")

    # CRITICAL ALIGNMENT: This assumes _smooth uses mode='same' and returns same length as input.
    # If your _smooth works differently (e.g. mode='valid' output from your original code was
    # len(input) + window_len - 1, which then needs trimming), adjust this section.
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    
    if len(y_smoothed) < 3:
        print("Smoothed data too short for processing.")
        return ([[], []], [[], []])

    # --- 2. Detect Initial Extrema Candidates ---
    initial_peaks_x, initial_peaks_y, \
    initial_valleys_x, initial_valleys_y = detect_initial_extrema_candidates(
        y_smoothed, x_coords_for_smoothed, deriv_method, first_deriv_zero_threshold_factor
    )

    # --- 3. First Alternation ---
    cand_peaks_x_alt, cand_peaks_y_alt, \
    cand_valleys_x_alt, cand_valleys_y_alt = _ensure_alternation(
        (initial_peaks_x, initial_peaks_y), (initial_valleys_x, initial_valleys_y)
    )

    # --- 4. Apply Clustering Filter ---
    (final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y,
     plot_data_peaks, plot_data_valleys) = apply_clustering_filter_to_extrema(
        cand_peaks_x_alt, cand_peaks_y_alt, cand_valleys_x_alt, cand_valleys_y_alt,
        dbscan_eps_peaks, dbscan_min_samples_peaks,
        dbscan_eps_valleys, dbscan_min_samples_valleys,
        min_features_for_clustering, min_extrema_for_spacing_model
    )
    # plot_data_peaks/valleys = (scaled_features, labels, feed_x, feed_y)
    
    extrema = refine_extrema_to_original_data(
        final_peaks_x, final_peaks_y,  # From process_spectrum_for_lfc_lines (on smoothed/rebinned scale)
        final_valleys_x, final_valleys_y,
        x_axis_orig, y_axis_orig,
        search_window_orig_pixels=0)
    final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y = extrema
    
    
    # --- 5. Plotting ---
    if plot:
        # candidate extrema (for plotting)
        cand_extrema = refine_extrema_to_original_data(
            plot_data_peaks[2], plot_data_peaks[3],  
            plot_data_valleys[2], plot_data_valleys[3],
            x_axis_orig, y_axis_orig,
            search_window_orig_pixels=0)
        cand_peaks_x, cand_peaks_y, cand_valleys_x, cand_valleys_y = cand_extrema
        if plot_spectrum_and_derivs_flag:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1) if y_smoothed.size > 0 else np.array([])
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2) if y_smoothed.size > 0 else np.array([])
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y, # Raw from derivatives
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        if plot_clustering_details_flag:
            # Candidates fed to clustering for peaks are plot_data_peaks[2] and plot_data_peaks[3]
            # Candidates fed to clustering for valleys are plot_data_valleys[2] and plot_data_valleys[3]
            plot_clustering_2d_results(
                x_axis_orig, y_axis_orig,
                x_coords_for_smoothed, y_smoothed,
                cand_peaks_x, cand_peaks_y, 
                cand_valleys_x, cand_valleys_y,
                final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y
            )
            if plot_data_peaks[0] is not None: # scaled_peak_features
                 plot_feature_space_3d(plot_data_peaks[0], plot_data_peaks[1], is_peak=True)
            if plot_data_valleys[0] is not None: # scaled_valley_features
                 plot_feature_space_3d(plot_data_valleys[0], plot_data_valleys[1], is_peak=False)

    
    return [final_peaks_x.tolist(), final_peaks_y.tolist()], \
           [final_valleys_x.tolist(), final_valleys_y.tolist()]

def process_spectrum_for_lfc_lines_v3( # Renamed
    y_axis_orig, x_axis_orig=None,
    # Detailed processing parameters
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    # Combined dict for clustering and its feature calculation settings
    clustering_and_feature_params: dict = None,
    # Plotting
    plot_main_details=False, # Single flag for all detailed plots now
    plot_spectrum_and_derivs_flag=True, # Sub-flags if plot_main_details is True
    plot_clustering_details_flag=True,  # Sub-flags
    # Refinement to original data
    refine_to_original_params: dict = None,
    verbose=False # Added verbose flag
):
    x_axis_orig, y_axis_orig = _datacheck(x_axis_orig, y_axis_orig) # Use renamed _datacheck

    # --- Set default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust','user_val':None,'gw_params':{'target_min_period_pixels':10,'target_max_period_pixels':20,'default_window_period':15,'overall_min_period_pixels':5}}
    
    global_noise_val = robust_noise_std(y_axis_orig - np.median(y_axis_orig)) # Calculate once
    if clustering_and_feature_params is None:
        clustering_and_feature_params = {
            'dbscan_eps':0.5,'dbscan_min_samples':5,
            'min_features_for_clustering':8, 'min_extrema_for_spacing_model':4,
            'global_noise_std_for_fallback': global_noise_val,
            'noise_estimation_window_pixels_feat':15,
            'spacing_poly_degree_feat':2, 'spacing_uncertainty_const_feat':2.0,
            'feature_scaling_options':{'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
        }
    elif 'global_noise_std_for_fallback' not in clustering_and_feature_params:
        clustering_and_feature_params['global_noise_std_for_fallback'] = global_noise_val

    if refine_to_original_params is None:
        refine_to_original_params = {'search_window_orig_pixels': 3}

    # --- Stage 1: Supersample, Smooth, Derivatives, Initial Candidates (on WHOLE spectrum) ---
    # This logic is from the old `process_spectrum_for_lfc_lines` or `process_single_lfc_segment`
    # but applied to the full y_axis_orig.
    if verbose: print("--- Stage 1: Initial Peak/Valley Detection on Full Spectrum ---")
    
    if super_sample_factor > 1:
        y_rebinned = _rebin(y_axis_orig, newshape=(int(super_sample_factor * len(y_axis_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_orig), np.max(x_axis_orig), len(y_rebinned))
    else:
        y_rebinned = np.copy(y_axis_orig); x_rebinned = np.copy(x_axis_orig)

    plot_gw = plot_main_details and plot_spectrum_and_derivs_flag
    win_method = window_len_method_options['method']
    gw_p = window_len_method_options.get('gw_params', {})
    if win_method == 'auto_robust':
        window_len_on_orig_scale = get_window_robust_targeted(y_axis_orig, plot=plot_gw, **gw_p)
    elif win_method == 'auto_original_get_window':
        window_len_on_orig_scale = get_window(y_axis_orig, plot=plot_gw)
    elif win_method == 'user_defined' and window_len_method_options.get('user_val') is not None:
        window_len_on_orig_scale = mathfunc.round_down_to_odd(window_len_method_options['user_val'])
    else:
        window_len_on_orig_scale = gw_p.get('default_window_period',15)
        if verbose: print(f"    Using default window {window_len_on_orig_scale} for full spectrum.")

    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Orig scale window: {window_len_on_orig_scale}, Rebinned smooth window: {actual_smoothing_window_len}")
    
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])

    deriv1 = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
    deriv2 = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
    cand_idx_full = []
    if deriv1.size >=2:
        deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1)) if deriv1.size > 0 else 1e-9
        sign_chg = np.diff(np.sign(deriv1)); cross_idx = np.where(np.abs(sign_chg)==2)[0]
        for idx in cross_idx:
            if idx+1 < deriv1.size: cand_idx_full.append(idx if np.abs(deriv1[idx]) < np.abs(deriv1[idx+1]) else idx+1)
            else: cand_idx_full.append(idx)
    cand_idx_full = np.array(sorted(list(set(cand_idx_full))), dtype=int)
    if cand_idx_full.size == 0: print("No initial candidates from derivatives."); return ([[],[]]),([[],[]])
    
    valid_cand_idx_full = cand_idx_full[(cand_idx_full < len(deriv2)) & (cand_idx_full < len(x_coords_for_smoothed))]
    if valid_cand_idx_full.size == 0: print("No valid initial candidates after bounds check."); return ([[],[]]),([[],[]])

    max_i = valid_cand_idx_full[deriv2[valid_cand_idx_full] < 0]
    min_i = valid_cand_idx_full[deriv2[valid_cand_idx_full] > 0]
    
    initial_peaks_x = x_coords_for_smoothed[max_i] if max_i.size > 0 else np.array([])
    initial_peaks_y = y_smoothed[max_i] if max_i.size > 0 else np.array([])
    initial_valleys_x = x_coords_for_smoothed[min_i] if min_i.size > 0 else np.array([])
    initial_valleys_y = y_smoothed[min_i] if min_i.size > 0 else np.array([])
    if verbose: print(f"  Initial derivative candidates: {len(initial_peaks_x)} peaks, {len(initial_valleys_x)} valleys.")

    # --- Stage 2: Apply Combined Clustering Filter to these initial candidates ---
    if verbose: print(f"\n--- Stage 2: Applying Combined Clustering Filter ---")
    (final_peaks_x_arr, final_peaks_y_arr,
     final_valleys_x_arr, final_valleys_y_arr,
     plot_data_combined) = apply_clustering_filter_to_extrema_v3(
        initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
        clustering_and_feature_params['dbscan_eps'],
        clustering_and_feature_params['dbscan_min_samples'],
        clustering_and_feature_params['min_features_for_clustering'],
        y_axis_orig, # Pass original y_axis for local noise estimation context
        x_axis_orig, # Pass original x_axis for local noise estimation context
        clustering_and_feature_params['global_noise_std_for_fallback'],
        clustering_and_feature_params['noise_estimation_window_pixels_feat'],
        clustering_and_feature_params['spacing_poly_degree_feat'],
        clustering_and_feature_params['spacing_uncertainty_const_feat'],
        clustering_and_feature_params['feature_scaling_options']
    )
    if verbose: print(f"  After clustering: {len(final_peaks_x_arr)} peaks, {len(final_valleys_x_arr)} valleys.")


    # --- Stage 3: Refine ALL clustered extrema to original data grid and flux ---
    if refine_to_original_params.get('search_window_orig_pixels', -1) >= 0:
        if verbose: print(f"\n--- Stage 3: Refining all {len(final_peaks_x_arr)}P / {len(final_valleys_x_arr)}V to original data ---")
        if 'refine_extrema_to_original_data' in globals(): # Check if function is defined
            r_p_x, r_p_y, r_v_x, r_v_y = refine_extrema_to_original_data(
                final_peaks_x_arr, final_peaks_y_arr, # Pass y from smoothed for now, refine_extrema ignores it
                final_valleys_x_arr, final_valleys_y_arr,
                x_axis_orig, y_axis_orig,
                search_window_orig_pixels=refine_to_original_params['search_window_orig_pixels']
            )
            final_peaks_x_arr, final_peaks_y_arr = r_p_x, r_p_y
            final_valleys_x_arr, final_valleys_y_arr = r_v_x, r_v_y
            if verbose: print(f"    After refinement: {len(final_peaks_x_arr)} peaks, {len(final_valleys_x_arr)} valleys.")
        else:
            print("Warning: refine_extrema_to_original_data function not found. Skipping refinement.")
            # Keep y-values from smoothed data if not refining
            # final_peaks_y_arr and final_valleys_y_arr are already populated

    # Final sort (alternation in apply_clustering_filter_to_extrema_v3 should handle most)
    if final_peaks_x_arr.size > 0:
        sort_p_idx = np.argsort(final_peaks_x_arr)
        final_peaks_x_arr, final_peaks_y_arr = final_peaks_x_arr[sort_p_idx], final_peaks_y_arr[sort_p_idx]
    if final_valleys_x_arr.size > 0:
        sort_v_idx = np.argsort(final_valleys_x_arr)
        final_valleys_x_arr, final_valleys_y_arr = final_valleys_x_arr[sort_v_idx], final_valleys_y_arr[sort_v_idx]

    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        # Unpack data for plotting from apply_clustering_filter_to_extrema_v3
        # plot_data_combined = (features_unscaled, scaled_features, labels, cand_x, cand_y, cand_types)
        cand_x_for_plot = plot_data_combined[3]
        cand_y_for_plot = plot_data_combined[4]
        cand_types_for_plot = plot_data_combined[5]

        if plot_spectrum_and_derivs_flag:
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1)) if deriv1.size > 0 else 0
            plot_smoothing_and_derivatives( # Needs to be defined
                x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
                deriv1, deriv2, # Pass calculated derivatives
                initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        if plot_clustering_details_flag:
            plot_clustering_2d_results( # Needs to be defined
                x_axis_orig, y_axis_orig, # Plot on original data as background
                x_coords_for_smoothed, y_smoothed, # Smoothed data for context if needed
                cand_x_for_plot[cand_types_for_plot==1], cand_y_for_plot[cand_types_for_plot==1], 
                cand_x_for_plot[cand_types_for_plot==-1], cand_y_for_plot[cand_types_for_plot==-1],
                final_peaks_x_arr, final_peaks_y_arr, final_valleys_x_arr, final_valleys_y_arr
            )
            if plot_data_combined[1] is not None: # scaled_features_combined
                 plot_feature_space_3d_v2( # Needs to be defined
                     scaled_features=plot_data_combined[1], # scaled_features_combined
                     labels=plot_data_combined[2],          # labels from DBSCAN
                     original_types=plot_data_combined[5],  # all_cand_types
                     title_suffix="Full Spectrum" # Or specific segment ID if called from process_single_lfc_segment_v3
                 )

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]

# ==============================================================================
# == SECTION 6: EXAMPLE USAGE (Illustrative) ===================================
# ==============================================================================
if __name__ == '__main__':
    # Create some dummy spectrum data
    np.random.seed(42)
    x_spec = np.linspace(0, 200, 2000)
    y_signal = (np.sin(x_spec / 3) * (1 + 0.5*np.sin(x_spec/50)) + # Modulated comb
                0.8 * np.sin(x_spec / 2.5 + np.pi/4) + # Another comb component
                0.1 * (x_spec / 100) ) # Blaze like trend
    y_noise = np.random.normal(0, 0.1, x_spec.size)
    y_supercontinuum = 0.5 * np.exp(-((x_spec - 100)**2) / (2 * 80**2)) + 0.2 # Broad supercontinuum
    y_spec = y_signal + y_noise + y_supercontinuum
    y_spec = (y_spec - np.min(y_spec)) * 2 # Ensure positive and some amplitude

    print(f"Running example with dummy data (length {len(y_spec)})...")

    # Call the main processing function
    # You'll need to tune these parameters extensively for real data
    peaks_result, valleys_result = process_spectrum_for_lfc_lines(
        y_spec, x_spec,
        super_sample_factor=3, # Lower for faster example run
        window_len_method='auto_robust',
        gw_min_period_pixels=5,       # Min expected period in ORIGINAL pixels
        gw_max_period_pixels=50,      # Max expected period in ORIGINAL pixels
        gw_default_window_period=15,
        first_deriv_zero_threshold_factor=0.03,
        dbscan_eps_peaks=0.4,        # TUNE (MinMaxScaler output is 0-1)
        dbscan_min_samples_peaks=4,  # TUNE
        dbscan_eps_valleys=0.4,      # TUNE
        dbscan_min_samples_valleys=4,# TUNE
        min_features_for_clustering=8,
        min_extrema_for_spacing_model=4,
        plot_main=True,
        plot_spectrum_and_derivs_flag=True,
        plot_clustering_details_flag=True
    )

    final_peaks_x, final_peaks_y = peaks_result
    final_valleys_x, final_valleys_y = valleys_result

    print(f"\nFound {len(final_peaks_x)} final peaks and {len(final_valleys_x)} final valleys.")

    # Minimal final plot
    plt.figure(figsize=(12,6))
    plt.plot(x_spec, y_spec, label="Original Spectrum", alpha=0.6, color='grey')
    if final_peaks_x:
        plt.scatter(final_peaks_x, final_peaks_y, color='red', marker='^', s=50, label="Final Peaks")
    if final_valleys_x:
        plt.scatter(final_valleys_x, final_valleys_y, color='blue', marker='v', s=50, label="Final Valleys")
    plt.title("Final Detected Extrema on Original Spectrum")
    plt.xlabel("X-coordinate")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()
    
    
# --- SECTION 2: SIGNAL REGION IDENTIFICATION FUNCTIONS ---

def robust_noise_std(y_data):
    """Estimates noise standard deviation using Median Absolute Deviation."""
    y_data = np.asarray(y_data)
    if y_data.size == 0:
        return 1e-6 # Avoid zero if empty
    # scale='normal' makes MAD comparable to Gaussian std
    mad_val = scipy.stats.median_abs_deviation(y_data, scale='normal')
    return mad_val if mad_val > 1e-9 else 1e-9 # Floor value

def get_sliding_window_features(y_detrended, win_size, step_size,
                                expected_lfc_period_min, expected_lfc_period_max,
                                use_spectral_feature=False): # Added flag for spectral
    """
    Calculates variance and kurtosis in sliding windows.
    Optionally calculates spectral peak strength (can be slow).
    Returns feature arrays and corresponding window center indices.
    """
    num_points = len(y_detrended)
    window_centers_idx = []
    variances = []
    kurtoses = []
    spectral_peak_strengths = [] if use_spectral_feature else None

    if use_spectral_feature:
        min_lfc_freq = 1.0 / expected_lfc_period_max if expected_lfc_period_max > 1e-9 else 0
        max_lfc_freq = 1.0 / expected_lfc_period_min if expected_lfc_period_min > 1e-9 else np.inf

    for i in range(0, num_points - win_size + 1, step_size):
        window_centers_idx.append(i + win_size // 2)
        y_window = y_detrended[i : i + win_size]
        
        if len(y_window) < 4: # Kurtosis needs at least 4 points for non-NaN typically
            variances.append(0)
            kurtoses.append(-3) # Default for too few points (Gaussian is 0 for Fisher)
            if use_spectral_feature: spectral_peak_strengths.append(0)
            continue

        variances.append(np.var(y_window))
        kurtoses.append(scipy.stats.kurtosis(y_window, fisher=True, bias=False))

        if use_spectral_feature:
            try:
                # Use a shorter nperseg if window is short, but not less than some minimum
                nperseg_welch = min(len(y_window), max(32, len(y_window)//2)) 
                if nperseg_welch < 4 : # welch needs nperseg to be reasonable
                    spectral_peak_strengths.append(0)
                    continue
                freqs, pxx = welch(y_window, fs=1.0, nperseg=nperseg_welch, scaling='density')
                lfc_band_mask = (freqs >= min_lfc_freq) & (freqs <= max_lfc_freq)
                if np.any(lfc_band_mask) and len(pxx[lfc_band_mask]) > 0:
                    spectral_peak_strengths.append(np.max(pxx[lfc_band_mask]))
                else:
                    spectral_peak_strengths.append(0)
            except ValueError: # e.g., if window too short for welch
                spectral_peak_strengths.append(0)

    result = [np.array(window_centers_idx), np.array(variances), np.array(kurtoses)]
    if use_spectral_feature:
        result.append(np.array(spectral_peak_strengths))
    return tuple(result)


def find_lfc_signal_regions(
    y_data_orig,
    segmentation_params: dict
):
    """
    Identifies regions containing LFC signal in a 1D array.
    Uses parameters from segmentation_params dictionary.
    """
    y_data = np.asarray(y_data_orig)
    # Unpack parameters with defaults
    expected_lfc_period = segmentation_params.get('expected_lfc_period_typical', 15)
    win_factor = segmentation_params.get('feature_win_size_factor', 4.0)
    step_factor = segmentation_params.get('feature_step_size_factor', 0.5)
    var_thresh_f = segmentation_params.get('variance_thresh_factor', 5.0)
    kur_thresh_min = segmentation_params.get('kurtosis_thresh_min', 0.5)
    snr_thresh = segmentation_params.get('segment_snr_thresh', 3.0)
    min_len_periods = segmentation_params.get('min_segment_len_periods', 3.0)
    bound_win_factor = segmentation_params.get('boundary_refine_window_factor', 0.5) # Factor of LFC period
    bound_var_drop_f = segmentation_params.get('boundary_variance_drop_factor', 2.0)
    plot_diag = segmentation_params.get('plot_diagnostics', False)
    use_spectral = segmentation_params.get('use_spectral_feature_for_segmentation', False)
    spectral_thresh_factor = segmentation_params.get('spectral_thresh_factor', 5.0) # e.g. X times noise floor PSD

    if y_data.ndim != 1 or len(y_data) < 2 * expected_lfc_period:
        print("Input data for segmentation is not 1D or too short.")
        return []

    # --- 1. Preprocessing and Noise Estimation ---
    y_detrended = y_data - np.median(y_data) # Simple global median detrend
    noise_sigma = robust_noise_std(y_detrended)
    noise_variance = noise_sigma**2

    # --- 2. Feature Extraction ---
    win_size = mathfunc.round_down_to_odd(int(expected_lfc_period * win_factor))
    win_size = max(5, win_size)
    step_size = mathfunc.round_down_to_odd(int(expected_lfc_period * step_factor))
    step_size = max(1, step_size)

    min_p = expected_lfc_period * segmentation_params.get('expected_lfc_period_min_factor', 0.7)
    max_p = expected_lfc_period * segmentation_params.get('expected_lfc_period_max_factor', 1.3)

    extracted_features = get_sliding_window_features(
        y_detrended, win_size, step_size, min_p, max_p, use_spectral_feature=use_spectral
    )
    window_centers_idx = extracted_features[0]
    variances = extracted_features[1]
    kurtoses = extracted_features[2]
    spectral_strengths = extracted_features[3] if use_spectral else None

    if window_centers_idx.size == 0: return []

    # --- 3. Candidate Region Identification ---
    variance_threshold = var_thresh_f * noise_variance
    var_signal_mask = variances > variance_threshold
    kur_signal_mask = kurtoses > kur_thresh_min
    
    combined_signal_mask = var_signal_mask & kur_signal_mask
    if use_spectral and spectral_strengths is not None:
        # Estimate noise floor for PSD if not already done, or use a relative threshold
        # For simplicity, let's assume spectral_thresh_factor is relative to mean of non-var/kur signal
        # This is a rough way to set spectral threshold
        noise_like_psd_mean = np.mean(spectral_strengths[~(var_signal_mask & kur_signal_mask)]) if np.any(~(var_signal_mask & kur_signal_mask)) else np.mean(spectral_strengths)
        spectral_threshold = spectral_thresh_factor * (noise_like_psd_mean if np.isfinite(noise_like_psd_mean) and noise_like_psd_mean > 0 else np.median(spectral_strengths))
        spec_signal_mask = spectral_strengths > spectral_threshold
        combined_signal_mask = combined_signal_mask & spec_signal_mask # More stringent if spectral is used

    candidate_segments_pixels = []
    if np.any(combined_signal_mask):
        diff_mask = np.diff(np.concatenate(([False], combined_signal_mask, [False])).astype(int))
        starts_win_mask = np.where(diff_mask == 1)[0]
        ends_win_mask = np.where(diff_mask == -1)[0]
        for s_wm_idx, e_wm_idx in zip(starts_win_mask, ends_win_mask):
            if e_wm_idx <= s_wm_idx : continue # Should not happen if mask is processed correctly
            seg_start = max(0, window_centers_idx[s_wm_idx] - win_size // 2)
            seg_end = min(len(y_data), window_centers_idx[e_wm_idx - 1] + win_size // 2)
            if seg_end > seg_start: candidate_segments_pixels.append((seg_start, seg_end))
    
    if not candidate_segments_pixels: return []

    candidate_segments_pixels.sort(key=lambda x: x[0])
    merged_segments = [candidate_segments_pixels[0]]
    for cur_s, cur_e in candidate_segments_pixels[1:]:
        prev_s, prev_e = merged_segments[-1]
        if cur_s <= prev_e + step_size: merged_segments[-1] = (prev_s, max(prev_e, cur_e))
        else: merged_segments.append((cur_s, cur_e))
    candidate_segments_pixels = merged_segments

    # --- 4. Refinement and Validation ---
    final_lfc_regions = []
    min_len_px = mathfunc.round_down_to_odd(min_len_periods * expected_lfc_period)
    boundary_win = mathfunc.round_down_to_odd(int(bound_win_factor * expected_lfc_period))
    boundary_win = max(3, boundary_win)


    for start_px, end_px in candidate_segments_pixels:
        if (end_px - start_px) < min_len_px: continue
        segment_data = y_detrended[start_px:end_px]
        if segment_data.size == 0: continue
        segment_snr = np.mean(segment_data**2) / noise_variance
        if segment_snr < snr_thresh: continue

        # Boundary Refinement
        refined_s, refined_e = start_px, end_px
        # Walk left
        for s_b in range(start_px, max(-1, start_px - win_size), -boundary_win): # Check up to one feature window away
            check_s, check_e = max(0, s_b - boundary_win // 2), max(0, s_b + boundary_win // 2) +1
            if check_s >= check_e or check_s < 0: break
            if np.var(y_detrended[check_s:check_e]) < bound_var_drop_f * noise_variance:
                refined_s = check_e; break # Noise starts after this window
            refined_s = check_s # Signal continues
        # Walk right
        for e_b in range(end_px -1 , min(len(y_data), end_px + win_size), boundary_win):
            check_s, check_e = min(len(y_data)-1, e_b - boundary_win // 2), min(len(y_data)-1, e_b + boundary_win // 2)+1
            if check_s >= check_e or check_e > len(y_data) : break
            if np.var(y_detrended[check_s:check_e]) < bound_var_drop_f * noise_variance:
                refined_e = check_s; break # Noise starts at this window
            refined_e = check_e # Signal continues

        if (refined_e - refined_s) >= min_len_px:
            final_lfc_regions.append((refined_s, refined_e))

    if not final_lfc_regions: return []
    final_lfc_regions.sort(key=lambda x: x[0])
    truly_final = [final_lfc_regions[0]]
    for cur_s, cur_e in final_lfc_regions[1:]:
        prev_s, prev_e = truly_final[-1]
        if cur_s <= prev_e : truly_final[-1] = (prev_s, max(prev_e, cur_e))
        else: truly_final.append((cur_s, cur_e))
    final_lfc_regions = truly_final
    
    if plot_diag: # Simplified plotting, expand as needed
        plt.figure(figsize=(12, 4))
        plt.plot(y_data, label='Original Data', alpha=0.6)
        for s, e in final_lfc_regions: plt.axvspan(s, e, color='lightgreen', alpha=0.5, label='LFC Region' if s==final_lfc_regions[0][0] else None)
        plt.title(f"LFC Segmentation Results (Noise std: {noise_sigma:.2f})")
        plt.legend(); plt.show()
        
    return final_lfc_regions


# --- SECTION 3: DETAILED PEAK/VALLEY DETECTION FOR A SINGLE SEGMENT ---

def process_single_lfc_segment(
    y_segment_data, x_segment_original_indices,
    original_full_x_axis,
    super_sample_factor,
    window_len_method_options: dict,
    deriv_method, first_deriv_zero_threshold_factor,
    dbscan_params: dict,
    min_features_for_clustering, min_extrema_for_spacing_model,
    plot_segment_details=False
):
    print(f"  Processing segment: orig_idx {x_segment_original_indices[0]}-{x_segment_original_indices[-1]}, len {len(y_segment_data)}")
    
    min_len_for_detailed = dbscan_params.get('min_lfc_period_for_smoothing', 10) * 3 # Heuristic
    if len(y_segment_data) < min_len_for_detailed:
        print(f"    Segment too short ({len(y_segment_data)} vs min {min_len_for_detailed}). Skipping detailed processing.")
        return (np.array([]), np.array([])), (np.array([]), np.array([]))

    # --- 1. Supersample the SEGMENT ---
    if super_sample_factor > 1:
        y_seg_rebinned = _rebin(y_segment_data, newshape=(int(super_sample_factor * len(y_segment_data)),))
        x_seg_rebinned = np.linspace(
            original_full_x_axis[x_segment_original_indices[0]],
            original_full_x_axis[x_segment_original_indices[-1]],
            len(y_seg_rebinned)
        )
    else:
        y_seg_rebinned = np.copy(y_segment_data)
        x_seg_rebinned = original_full_x_axis[x_segment_original_indices]

    # --- 2. Determine Smoothing Window for the SEGMENT ---
    win_method = window_len_method_options['method']
    gw_p = window_len_method_options.get('gw_params', {}) # Ensure gw_params exists
    if win_method == 'auto_robust':
        window_len_on_orig_scale_seg = get_window_robust_targeted(
            y_segment_data, plot=False, # Avoid too many plots here, master flag controls main one
            target_min_period_pixels=gw_p.get('target_min_period_pixels', 10),
            target_max_period_pixels=gw_p.get('target_max_period_pixels', 20),
            default_window_period=gw_p.get('default_window_period',15),
            overall_min_period_pixels=gw_p.get('overall_min_period_pixels',5)
        )
    elif win_method == 'user_defined' and window_len_method_options.get('user_val') is not None:
        window_len_on_orig_scale_seg = mathfunc.round_down_to_odd(window_len_method_options['user_val'])
    else: # Fallback
        window_len_on_orig_scale_seg = gw_p.get('default_window_period',15)
        print(f"    Using default window {window_len_on_orig_scale_seg} for segment.")


    actual_smoothing_window_len = mathfunc.round_down_to_odd(
        int(window_len_on_orig_scale_seg * super_sample_factor)
    )
    if len(y_seg_rebinned) > 0: # Check if y_seg_rebinned is not empty
        max_possible_win_len = mathfunc.round_down_to_odd(len(y_seg_rebinned) // 2 -1 if len(y_seg_rebinned) // 2 -1 >=1 else 1)
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_possible_win_len)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    
    # --- 3. Smooth SEGMENT and Align (CRITICAL) ---
    y_seg_smoothed = _smooth(y_seg_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_seg_smoothed = x_seg_rebinned # Assumes _smooth output is same length and aligned

    if len(y_seg_smoothed) < 3: return (np.array([]), np.array([])), (np.array([]), np.array([]))

    # --- 4. Derivatives and Initial Candidates on SEGMENT ---
    deriv1 = mathfunc.derivative1d(y_seg_smoothed, order=1, method=deriv_method)
    deriv2 = mathfunc.derivative1d(y_seg_smoothed, order=2, method=deriv_method)
    
    cand_idx_seg = []
    if deriv1.size >=2: # Check for sufficient elements for diff
        deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1)) if deriv1.size > 0 else 1e-9
        sign_chg = np.diff(np.sign(deriv1)); cross_idx = np.where(np.abs(sign_chg)==2)[0]
        for idx in cross_idx:
            if idx+1 < deriv1.size: cand_idx_seg.append(idx if np.abs(deriv1[idx]) < np.abs(deriv1[idx+1]) else idx+1)
            else: cand_idx_seg.append(idx) # If idx is the last possible point for crossing
    cand_idx_seg = np.array(sorted(list(set(cand_idx_seg))), dtype=int)
    
    if cand_idx_seg.size == 0: return (np.array([]),np.array([])), (np.array([]),np.array([]))
    
    valid_cand_idx_seg = cand_idx_seg[(cand_idx_seg < len(deriv2)) & (cand_idx_seg < len(x_coords_for_seg_smoothed))]
    if valid_cand_idx_seg.size == 0: return (np.array([]),np.array([])), (np.array([]),np.array([]))

    max_i = valid_cand_idx_seg[deriv2[valid_cand_idx_seg] < 0]
    min_i = valid_cand_idx_seg[deriv2[valid_cand_idx_seg] > 0]
    
    seg_peaks_x_init = x_coords_for_seg_smoothed[max_i] if max_i.size > 0 else np.array([])
    seg_peaks_y_init = y_seg_smoothed[max_i] if max_i.size > 0 else np.array([])
    seg_valleys_x_init = x_coords_for_seg_smoothed[min_i] if min_i.size > 0 else np.array([])
    seg_valleys_y_init = y_seg_smoothed[min_i] if min_i.size > 0 else np.array([])

    # --- 5. Alternation and DBSCAN for the SEGMENT ---
    cand_p_x_alt, cand_p_y_alt, cand_v_x_alt, cand_v_y_alt = _ensure_alternation(
        seg_peaks_x_init, seg_peaks_y_init, seg_valleys_x_init, seg_valleys_y_init
    )
    
    final_p_x, final_p_y = cand_p_x_alt, cand_p_y_alt # Start with alternated candidates
    if cand_p_x_alt.size >= min_features_for_clustering and cand_v_x_alt.size > 0:
        p_feat_unscaled = _calculate_features_for_extrema(cand_p_x_alt, cand_p_y_alt, cand_v_x_alt, cand_v_y_alt, True, min_extrema_for_spacing_model=min_extrema_for_spacing_model)
        if p_feat_unscaled.size > 0 and p_feat_unscaled.shape[0] > 0 : # Check if features were actually calculated
            p_feat_scaled = RobustScaler().fit_transform(p_feat_unscaled)
            final_p_x, final_p_y, _ = filter_extrema_with_dbscan(cand_p_x_alt, cand_p_y_alt, p_feat_scaled, dbscan_params['eps_peaks'], dbscan_params['min_samples_peaks'])

    p_after_filt, py_after_filt, v_for_vfilt, vy_for_vfilt = _ensure_alternation(final_p_x, final_p_y, cand_v_x_alt, cand_v_y_alt)
    
    final_v_x, final_v_y = v_for_vfilt, vy_for_vfilt # Start with alternated candidates
    if v_for_vfilt.size >= min_features_for_clustering and p_after_filt.size > 0:
        v_feat_unscaled = _calculate_features_for_extrema(v_for_vfilt, vy_for_vfilt, p_after_filt, py_after_filt, False, min_extrema_for_spacing_model=min_extrema_for_spacing_model)
        if v_feat_unscaled.size > 0 and v_feat_unscaled.shape[0] > 0:
            v_feat_scaled = RobustScaler().fit_transform(v_feat_unscaled)
            final_v_x, final_v_y, _ = filter_extrema_with_dbscan(v_for_vfilt, vy_for_vfilt, v_feat_scaled, dbscan_params['eps_valleys'], dbscan_params['min_samples_valleys'])
            
    final_p_x_seg, final_p_y_seg, final_v_x_seg, final_v_y_seg = _ensure_alternation(final_p_x, final_p_y, final_v_x, final_v_y)
    
    if plot_segment_details:
        plt.figure(figsize=(10,4))
        plt.plot(x_coords_for_seg_smoothed, y_seg_smoothed, label=f'Smoothed Seg {x_segment_original_indices[0]}-{x_segment_original_indices[-1]}', alpha=0.7)
        if final_p_x_seg.size > 0: plt.scatter(final_p_x_seg, final_p_y_seg, color='red', marker='^', label='Peaks')
        if final_v_x_seg.size > 0: plt.scatter(final_v_x_seg, final_v_y_seg, color='green', marker='v', label='Valleys')
        plt.legend(); plt.title(f'Processed Segment (Orig Idx {x_segment_original_indices[0]} to {x_segment_original_indices[-1]})'); plt.show()

    return (final_p_x_seg, final_p_y_seg), (final_v_x_seg, final_v_y_seg)


# --- SECTION 4: MAIN ORCHESTRATING FUNCTION ---
def process_spectrum_for_lfc_lines_with_segmentation(
    y_axis_orig, x_axis_orig=None,
    segmentation_params: dict = None,
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    dbscan_params: dict = None,
    min_features_for_clustering=8, min_extrema_for_spacing_model=4,
    plot_main_segmentation=False, plot_segment_processing_details=False
):
    x_axis_orig, y_axis_orig = _datacheck(x_axis_orig, y_axis_orig)

    # --- Set default parameters if None ---
    if segmentation_params is None:
        segmentation_params = {
            'expected_lfc_period_typical': 15, 'feature_win_size_factor': 4.0,
            'feature_step_size_factor': 0.5, 'variance_thresh_factor': 4.0,
            'kurtosis_thresh_min': 0.3, 'segment_snr_thresh': 3.0,
            'min_segment_len_periods': 2.5, 
            'boundary_refine_window_factor': 0.5, # Factor of typical period
            'boundary_variance_drop_factor': 1.5, 
            'plot_diagnostics': plot_main_segmentation, # Controlled by main flag
            'use_spectral_feature_for_segmentation': False # Default to not use complex spectral feature
        }
    # Ensure plot_diagnostics in segmentation_params reflects master plot_main_segmentation
    segmentation_params['plot_diagnostics'] = plot_main_segmentation

    if window_len_method_options is None:
        window_len_method_options = {
            'method': 'auto_robust', 'user_val': None,
            'gw_params': {'target_min_period_pixels': 10, 'target_max_period_pixels': 20,
                          'default_window_period':15, 'overall_min_period_pixels':5}
        }
    if dbscan_params is None:
        dbscan_params = {
            'eps_peaks': 0.4, 'min_samples_peaks': 4,
            'eps_valleys': 0.4, 'min_samples_valleys': 4,
            'min_lfc_period_for_smoothing': segmentation_params.get('expected_lfc_period_typical',15)
        }

    # --- Pass 1: Identify LFC Signal Regions ---
    print("--- Stage 1: Identifying LFC Signal Regions ---")
    lfc_regions_indices = find_lfc_signal_regions(y_axis_orig, segmentation_params=segmentation_params)

    if not lfc_regions_indices:
        print("No LFC signal regions identified by segmentation. No peaks/valleys extracted.")
        return ([[], []], [[], []])

    all_final_peaks_x, all_final_peaks_y = [], []
    all_final_valleys_x, all_final_valleys_y = [], []

    print(f"\n--- Stage 2: Detailed Peak/Valley Detection in {len(lfc_regions_indices)} Identified Region(s) ---")
    for i, (start_idx, end_idx) in enumerate(lfc_regions_indices):
        y_segment_data = y_axis_orig[start_idx:end_idx]
        x_segment_original_indices = np.arange(start_idx, end_idx)

        (peaks_x_seg, peaks_y_seg), (valleys_x_seg, valleys_y_seg) = \
            process_single_lfc_segment(
                y_segment_data, x_segment_original_indices,
                x_axis_orig, super_sample_factor, window_len_method_options,
                deriv_method, first_deriv_zero_threshold_factor,
                dbscan_params, min_features_for_clustering, min_extrema_for_spacing_model,
                plot_segment_details=plot_segment_processing_details
            )
        
        if peaks_x_seg.size > 0:
            all_final_peaks_x.extend(peaks_x_seg.tolist())
            all_final_peaks_y.extend(peaks_y_seg.tolist())
        if valleys_x_seg.size > 0:
            all_final_valleys_x.extend(valleys_x_seg.tolist())
            all_final_valleys_y.extend(valleys_y_seg.tolist())

    final_peaks_x_arr = np.array(all_final_peaks_x)
    final_peaks_y_arr = np.array(all_final_peaks_y)
    final_valleys_x_arr = np.array(all_final_valleys_x)
    final_valleys_y_arr = np.array(all_final_valleys_y)

    # Sort combined results (essential if segments could produce out-of-order x values)
    if final_peaks_x_arr.size > 0:
        sort_p_idx = np.argsort(final_peaks_x_arr)
        final_peaks_x_arr, final_peaks_y_arr = final_peaks_x_arr[sort_p_idx], final_peaks_y_arr[sort_p_idx]
    if final_valleys_x_arr.size > 0:
        sort_v_idx = np.argsort(final_valleys_x_arr)
        final_valleys_x_arr, final_valleys_y_arr = final_valleys_x_arr[sort_v_idx], final_valleys_y_arr[sort_v_idx]
        
    # Optional: A final _ensure_alternation on the combined list if strict global alternation is critical
    # final_peaks_x_arr, final_peaks_y_arr, final_valleys_x_arr, final_valleys_y_arr = \
    # _ensure_alternation(final_peaks_x_arr, final_peaks_y_arr, final_valleys_x_arr, final_valleys_y_arr)


    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys across LFC regions.")

    if plot_main_segmentation:
        plt.figure(figsize=(15,6))
        plt.plot(x_axis_orig, y_axis_orig, label="Original Full Spectrum", alpha=0.5, color='grey')
        if lfc_regions_indices: # Check if list is not empty before accessing first element
            for s,e in lfc_regions_indices:
                 plt.axvspan(x_axis_orig[s], x_axis_orig[e], color='lightyellow', alpha=0.3, 
                             label='Identified LFC Region(s)' if s == lfc_regions_indices[0][0] else None)
        if final_peaks_x_arr.size > 0:
            plt.scatter(final_peaks_x_arr, final_peaks_y_arr, color='red', marker='^', s=30, label="Final Peaks")
        if final_valleys_x_arr.size > 0:
            plt.scatter(final_valleys_x_arr, final_valleys_y_arr, color='green', marker='v', s=30, label="Final Valleys")
        plt.title("Final Extrema from Identified LFC Regions"); plt.xlabel("X-coordinate"); plt.ylabel("Flux")
        plt.legend(); plt.grid(True, linestyle=':'); plt.show()

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]