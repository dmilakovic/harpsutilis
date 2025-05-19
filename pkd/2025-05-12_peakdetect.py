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

def robust_noise_std(y_data):
    """Estimates noise standard deviation using Median Absolute Deviation."""
    y_data = np.asarray(y_data)
    if y_data.size == 0:
        return 1e-6 # Avoid zero if empty
    # scale='normal' makes MAD comparable to Gaussian std
    mad_val = scipy.stats.median_abs_deviation(y_data, scale='normal')
    return mad_val if mad_val > 1e-9 else 1e-9 # Floor value

def robust_local_noise_std(y_data, center_idx, window_half_width, fallback_noise_std):
    """Estimates local noise std around center_idx, excluding a central region."""
    if y_data is None or len(y_data) == 0: return fallback_noise_std
    exclude_half_width = window_half_width // 3
    left_s=max(0,center_idx-window_half_width); left_e=max(0,center_idx-exclude_half_width)
    right_s=min(len(y_data),center_idx+exclude_half_width+1); right_e=min(len(y_data),center_idx+window_half_width+1)
    noise_samp=[];
    if left_e>left_s: noise_samp.extend(y_data[left_s:left_e])
    if right_e>right_s: noise_samp.extend(y_data[right_s:right_e])
    if len(noise_samp)<5: return fallback_noise_std # Need enough samples for MAD
    # Use nan_policy='omit' if y_data might contain NaNs
    mad_val=scipy.stats.median_abs_deviation(noise_samp,scale='normal', nan_policy='omit')
    return mad_val if mad_val>1e-9 else fallback_noise_std # Return fallback if MAD is effectively zero


def _fit_spacing_polynomial(extrema_x_coords, poly_degree=2):
    """Fits a polynomial S_exp(x_midpoint) = a*x_mid^2 + b*x_mid + c."""
    extrema_x_coords = np.asarray(extrema_x_coords)
    if len(extrema_x_coords) < poly_degree + 2 : return None, None # Need enough points
    spacings = np.diff(extrema_x_coords)
    if spacings.size == 0: return None, None # Need at least one spacing
    x_midpoints = (extrema_x_coords[:-1] + extrema_x_coords[1:]) / 2.0
    if len(x_midpoints) < poly_degree + 1: return None, None

    poly_features = PolynomialFeatures(degree=poly_degree, include_bias=True)
    try:
        X_poly = poly_features.fit_transform(x_midpoints.reshape(-1, 1))
    except ValueError: return None, None # If x_midpoints is problematic

    # Choose robust regressor based on number of points
    if len(X_poly) > X_poly.shape[1] * 2 and len(X_poly) > 5 :
        model = RANSACRegressor(random_state=42,
                                min_samples=max(X_poly.shape[1]+1, int(len(X_poly)*0.2)), # Ensure enough samples for model
                                residual_threshold=np.std(spacings)*0.75 if spacings.std() > 1e-6 else 1.0)
    else: model = HuberRegressor(epsilon=1.35)
    try: model.fit(X_poly, spacings); return model, poly_features
    except ValueError: return None, None


def _calculate_features_for_extrema_v3(
    all_extrema_x, all_extrema_y, all_extrema_types,
    original_y_data_for_noise, original_x_data_for_noise,
    noise_estimation_window_pixels, global_fallback_noise_std,
    spacing_poly_degree=2, spacing_uncertainty_const=2.0
):
    num_total_extrema = len(all_extrema_x)
    if num_total_extrema == 0: return np.array([]).reshape(0, 3)

    feature_x_coords = np.asarray(all_extrema_x)
    feature_prom_depth_snr = np.zeros(num_total_extrema)
    feature_spacing_dev_norm = np.zeros(num_total_extrema)
    raw_prom_depth_values = np.zeros(num_total_extrema)

    for i in range(num_total_extrema):
        x_i, y_i, type_i = all_extrema_x[i], all_extrema_y[i], all_extrema_types[i]
        opposite_type_mask = (all_extrema_types == -type_i)
        y_brackets = []
        if np.any(opposite_type_mask):
            opposite_x_cand = all_extrema_x[opposite_type_mask]; opposite_y_cand = all_extrema_y[opposite_type_mask]
            left_opp_indices = np.where(opposite_x_cand < x_i)[0]
            right_opp_indices = np.where(opposite_x_cand > x_i)[0]
            if left_opp_indices.size > 0:
                closest_left_idx_in_opposites = left_opp_indices[np.argmax(opposite_x_cand[left_opp_indices])]
                y_brackets.append(opposite_y_cand[closest_left_idx_in_opposites])
            if right_opp_indices.size > 0:
                closest_right_idx_in_opposites = right_opp_indices[np.argmin(opposite_x_cand[right_opp_indices])]
                y_brackets.append(opposite_y_cand[closest_right_idx_in_opposites])
        if y_brackets:
            raw_pd = (y_i - np.max(y_brackets)) if type_i == 1 else (np.min(y_brackets) - y_i)
            raw_prom_depth_values[i] = max(0, raw_pd)
        
        local_noise = global_fallback_noise_std
        if original_x_data_for_noise is not None and original_x_data_for_noise.size > 0:
            center_idx_in_orig = np.argmin(np.abs(original_x_data_for_noise - x_i))
            if original_y_data_for_noise is not None:
                local_noise = robust_local_noise_std(original_y_data_for_noise, center_idx_in_orig, 
                                                     noise_estimation_window_pixels // 2, global_fallback_noise_std)
        feature_prom_depth_snr[i] = raw_prom_depth_values[i] / local_noise if local_noise > 1e-9 else raw_prom_depth_values[i] / 1e-9

    sorted_indices_spacing = np.argsort(all_extrema_x)
    all_extrema_x_sorted = all_extrema_x[sorted_indices_spacing]
    spacing_model, spacing_poly_transformer = _fit_spacing_polynomial(all_extrema_x_sorted, poly_degree=spacing_poly_degree)

    if spacing_model is not None:
        temp_spacing_dev_norm_sorted = np.zeros(num_total_extrema)
        for i_sorted in range(num_total_extrema):
            x_i_s = all_extrema_x_sorted[i_sorted]; devs_norm_list = []
            if i_sorted > 0:
                s_bwd_act = x_i_s - all_extrema_x_sorted[i_sorted-1]; x_mid_bwd = (x_i_s + all_extrema_x_sorted[i_sorted-1])/2.0
                s_bwd_exp = spacing_model.predict(spacing_poly_transformer.transform(np.array([[x_mid_bwd]])))[0]
                devs_norm_list.append((s_bwd_act - s_bwd_exp) / spacing_uncertainty_const)
            if i_sorted < num_total_extrema - 1:
                s_fwd_act = all_extrema_x_sorted[i_sorted+1] - x_i_s; x_mid_fwd = (x_i_s + all_extrema_x_sorted[i_sorted+1])/2.0
                s_fwd_exp = spacing_model.predict(spacing_poly_transformer.transform(np.array([[x_mid_fwd]])))[0]
                devs_norm_list.append((s_fwd_act - s_fwd_exp) / spacing_uncertainty_const)
            if devs_norm_list: temp_spacing_dev_norm_sorted[i_sorted] = np.mean(np.abs(devs_norm_list))
        original_order_indices = np.argsort(sorted_indices_spacing)
        feature_spacing_dev_norm = temp_spacing_dev_norm_sorted[original_order_indices]
    else: print("    Warning: Spacing model fit failed. Spacing deviation feature will be zero.")
    return np.vstack([feature_x_coords, feature_prom_depth_snr, feature_spacing_dev_norm]).T

def _calculate_features_for_one_type( # NEW: Calculates features for one type (P or V)
    extrema_x_of_type, extrema_y_of_type_refined, # Extrema we are calculating features for
    opposite_extrema_x, opposite_extrema_y_refined, # Their opposites for prom/depth
    is_peak_type, # Boolean: True if extrema_x_of_type are peaks
    original_y_data_for_noise, original_x_data_for_noise,
    noise_estimation_window_pixels, global_fallback_noise_std,
    spacing_poly_degree=2, spacing_uncertainty_const=2.0
):
    num_extrema = len(extrema_x_of_type)
    if num_extrema == 0: return np.array([]).reshape(0, 3)

    feature_x_coords = np.asarray(extrema_x_of_type)
    feature_prom_depth_snr = np.zeros(num_extrema)
    feature_spacing_dev_norm = np.zeros(num_extrema)

    # Calculate Prominence/Depth S/N
    for i in range(num_extrema):
        x_i, y_i_refined = extrema_x_of_type[i], extrema_y_of_type_refined[i]
        
        raw_pd_val = 0
        if opposite_extrema_x.size > 0: # Check if opposites exist
            left_opp_indices = np.where(opposite_extrema_x < x_i)[0]
            right_opp_indices = np.where(opposite_extrema_x > x_i)[0]
            y_brackets_refined = []
            if left_opp_indices.size > 0:
                closest_left_idx = left_opp_indices[np.argmax(opposite_extrema_x[left_opp_indices])]
                y_brackets_refined.append(opposite_extrema_y_refined[closest_left_idx])
            if right_opp_indices.size > 0:
                closest_right_idx = right_opp_indices[np.argmin(opposite_extrema_x[right_opp_indices])]
                y_brackets_refined.append(opposite_extrema_y_refined[closest_right_idx])
            
            if y_brackets_refined:
                raw_pd_val = (y_i_refined - np.max(y_brackets_refined)) if is_peak_type else \
                             (np.min(y_brackets_refined) - y_i_refined)
                raw_pd_val = max(0, raw_pd_val)
        
        local_noise = global_fallback_noise_std
        if original_x_data_for_noise is not None and original_x_data_for_noise.size > 0:
            center_idx_in_orig = np.argmin(np.abs(original_x_data_for_noise - x_i))
            if original_y_data_for_noise is not None:
                local_noise = robust_local_noise_std(original_y_data_for_noise, center_idx_in_orig, 
                                                     noise_estimation_window_pixels // 2, global_fallback_noise_std)
        feature_prom_depth_snr[i] = raw_pd_val / local_noise if local_noise > 1e-9 else raw_pd_val / 1e-9

    # Calculate Spacing Deviation (using only extrema_x_of_type)
    spacing_model, spacing_poly_transformer = _fit_spacing_polynomial(extrema_x_of_type, poly_degree=spacing_poly_degree)
    if spacing_model is not None:
        for i in range(num_extrema):
            x_i = extrema_x_of_type[i]; devs_norm_list = []
            if i > 0:
                s_bwd_act = x_i - extrema_x_of_type[i-1]; x_mid_bwd = (x_i + extrema_x_of_type[i-1])/2.0
                s_bwd_exp_arr = predict_spacing(x_mid_bwd, spacing_model, spacing_poly_transformer)
                if s_bwd_exp_arr is not None and np.isfinite(s_bwd_exp_arr[0]): devs_norm_list.append(abs(s_bwd_act - s_bwd_exp_arr[0]) / spacing_uncertainty_const)
            if i < num_extrema - 1:
                s_fwd_act = extrema_x_of_type[i+1] - x_i; x_mid_fwd = (x_i + extrema_x_of_type[i+1])/2.0
                s_fwd_exp_arr = predict_spacing(x_mid_fwd, spacing_model, spacing_poly_transformer)
                if s_fwd_exp_arr is not None and np.isfinite(s_fwd_exp_arr[0]): devs_norm_list.append(abs(s_fwd_act - s_fwd_exp_arr[0]) / spacing_uncertainty_const)
            if devs_norm_list: feature_spacing_dev_norm[i] = np.mean(np.abs(devs_norm_list))
    else: print(f"    Warning: Spacing model fit failed for {'peaks' if is_peak_type else 'valleys'}. Spacing dev will be zero.")
    return np.vstack([feature_x_coords, feature_prom_depth_snr, feature_spacing_dev_norm]).T


def _ensure_alternation(peaks_xy_tuple, valleys_xy_tuple):
    """
    Ensures strict P-V-P-V alternation from separated peak/valley inputs.
    Input: peaks_xy_tuple = (peaks_x_array, peaks_y_array)
           valleys_xy_tuple = (valleys_x_array, valleys_y_array)
    Output: (alt_peaks_x, alt_peaks_y, alt_valleys_x, alt_valleys_y)
    """
    # Unpack inputs, handling potential empty lists/arrays gracefully
    if peaks_xy_tuple and len(peaks_xy_tuple) == 2 and hasattr(peaks_xy_tuple[0], '__len__'):
        peaks_x_np, peaks_y_np = np.asarray(peaks_xy_tuple[0]), np.asarray(peaks_xy_tuple[1])
    else:
        peaks_x_np, peaks_y_np = np.array([]), np.array([])

    if valleys_xy_tuple and len(valleys_xy_tuple) == 2 and hasattr(valleys_xy_tuple[0], '__len__'):
        valleys_x_np, valleys_y_np = np.asarray(valleys_xy_tuple[0]), np.asarray(valleys_xy_tuple[1])
    else:
        valleys_x_np, valleys_y_np = np.array([]), np.array([])

    if not peaks_x_np.size and not valleys_x_np.size:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if not peaks_x_np.size:
        return np.array([]), np.array([]), valleys_x_np, valleys_y_np
    if not valleys_x_np.size:
        return peaks_x_np, peaks_y_np, np.array([]), np.array([])

    # Combine all features with a type indicator (1 for peak, -1 for valley)
    all_x = np.concatenate((peaks_x_np, valleys_x_np))
    all_y = np.concatenate((peaks_y_np, valleys_y_np))
    all_types = np.concatenate((np.ones(peaks_x_np.size), -np.ones(valleys_x_np.size)))

    # Sort by x-coordinate
    sorted_indices = np.argsort(all_x)
    sorted_x = all_x[sorted_indices]
    sorted_y = all_y[sorted_indices]
    sorted_types = all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list = [], [], []

    if not sorted_x.size: # Should not happen if checks above passed
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Add the first feature
    final_x_list.append(sorted_x[0])
    final_y_list.append(sorted_y[0])
    final_types_list.append(sorted_types[0])

    for i in range(1, sorted_x.size):
        current_x, current_y, current_type = sorted_x[i], sorted_y[i], sorted_types[i]
        last_x, last_y, last_type = final_x_list[-1], final_y_list[-1], final_types_list[-1]

        if current_type != last_type: # Different type, so alternate; add it
            final_x_list.append(current_x)
            final_y_list.append(current_y)
            final_types_list.append(current_type)
        else: # Same type as previous (e.g., P-P or V-V)
            if current_type == 1: # Both are peaks
                if current_y > last_y: # Current peak is higher, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
            else: # Both are valleys
                if current_y < last_y: # Current valley is lower, replace previous
                    final_x_list[-1], final_y_list[-1] = current_x, current_y
    
    final_x_arr = np.array(final_x_list)
    final_y_arr = np.array(final_y_list)
    final_types_arr = np.array(final_types_list)

    # Separate back into peaks and valleys
    out_peaks_x = final_x_arr[final_types_arr == 1]
    out_peaks_y = final_y_arr[final_types_arr == 1]
    out_valleys_x = final_x_arr[final_types_arr == -1]
    out_valleys_y = final_y_arr[final_types_arr == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y

def _ensure_alternation_and_tag(peaks_x, peaks_y, valleys_x, valleys_y):
    """
    Ensures strict P-V-P-V alternation AND returns a combined list with type tags,
    as well as separated lists.
    Type tags: +1 for peak, -1 for valley.
    """
    peaks_x_np, peaks_y_np = np.asarray(peaks_x), np.asarray(peaks_y)
    valleys_x_np, valleys_y_np = np.asarray(valleys_x), np.asarray(valleys_y)

    if not peaks_x_np.size and not valleys_x_np.size:
        return np.array([]), np.array([]), np.array([]), np.array([]), \
               np.array([]), np.array([]), np.array([])

    all_x_coords, all_y_coords, all_types_list = [], [], []
    if peaks_x_np.size > 0:
        all_x_coords.extend(peaks_x_np); all_y_coords.extend(peaks_y_np)
        all_types_list.extend([1] * peaks_x_np.size)
    if valleys_x_np.size > 0:
        all_x_coords.extend(valleys_x_np); all_y_coords.extend(valleys_y_np)
        all_types_list.extend([-1] * valleys_x_np.size)
    
    all_x, all_y, all_types = np.array(all_x_coords), np.array(all_y_coords), np.array(all_types_list)

    if not all_x.size:
        return np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

    sorted_indices = np.argsort(all_x)
    sorted_x, sorted_y, sorted_types = all_x[sorted_indices], all_y[sorted_indices], all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list_out = [], [], []
    if sorted_x.size > 0:
        final_x_list.append(sorted_x[0]); final_y_list.append(sorted_y[0]); final_types_list_out.append(sorted_types[0])
        for i in range(1, sorted_x.size):
            cx,cy,ct = sorted_x[i],sorted_y[i],sorted_types[i]
            lt = final_types_list_out[-1]
            if ct != lt:
                final_x_list.append(cx); final_y_list.append(cy); final_types_list_out.append(ct)
            else:
                if ct == 1 and cy > final_y_list[-1]: final_x_list[-1],final_y_list[-1]=cx,cy
                elif ct == -1 and cy < final_y_list[-1]: final_x_list[-1],final_y_list[-1]=cx,cy
    
    combined_x_alt = np.array(final_x_list)
    combined_y_alt = np.array(final_y_list)
    combined_types_alt = np.array(final_types_list_out)
    
    out_peaks_x = combined_x_alt[combined_types_alt == 1]
    out_peaks_y = combined_y_alt[combined_types_alt == 1]
    out_valleys_x = combined_x_alt[combined_types_alt == -1]
    out_valleys_y = combined_y_alt[combined_types_alt == -1]
    
    return out_peaks_x, out_peaks_y, out_valleys_x, out_valleys_y, \
           combined_x_alt, combined_y_alt, combined_types_alt


def fit_envelope_interp_smooth(
    extrema_x, # Sorted x-coordinates of peaks or valleys
    extrema_y, # Corresponding y-values (refined to original data)
    full_x_axis, # The x-axis for which the envelope should be defined (e.g., original_x_axis)
    smooth_window_len_pixels=51,
    smooth_window_type='nuttall'
):
    """
    Estimates an envelope by:
    1. Linearly interpolating between the given extrema points.
    2. Smoothing the resulting interpolated function.

    Args:
        extrema_x (np.array): X-coordinates of the extrema (peaks or valleys), must be sorted.
        extrema_y (np.array): Y-values of the extrema.
        full_x_axis (np.array): The complete x-axis over which to calculate the envelope.
        smooth_window_len_pixels (int): Window length for smoothing the interpolated envelope.
                                        Should be odd.
        smooth_window_type (str): Type of window for smoothing (e.g., 'nuttall').

    Returns:
        np.array: The smoothed envelope y-values corresponding to full_x_axis.
                  Returns None if input is insufficient.
    """
    extrema_x = np.asarray(extrema_x)
    extrema_y = np.asarray(extrema_y)
    full_x_axis = np.asarray(full_x_axis)

    if len(extrema_x) < 2: # Need at least two points for interpolation
        # print("Warning: Not enough extrema to create interpolated envelope. Returning flat line or NaNs.")
        # Fallback: could be a constant (median of extrema_y) or NaNs
        if len(extrema_y) > 0:
            return np.full_like(full_x_axis, np.median(extrema_y), dtype=float)
        else:
            return np.full_like(full_x_axis, np.nan, dtype=float)

    # Ensure extrema_x are sorted for np.interp
    sort_indices = np.argsort(extrema_x)
    sorted_extrema_x = extrema_x[sort_indices]
    sorted_extrema_y = extrema_y[sort_indices]
    
    # Prevent issues if full_x_axis extends beyond sorted_extrema_x range by specifying left/right fill
    # Use the y-value of the first/last extremum for extrapolation
    jagged_envelope = np.interp(
        full_x_axis,
        sorted_extrema_x,
        sorted_extrema_y,
        left=sorted_extrema_y[0], # Fill value for x < min(sorted_extrema_x)
        right=sorted_extrema_y[-1] # Fill value for x > max(sorted_extrema_x)
    )

    # Ensure smooth_window_len is odd and appropriate
    # window_len = mathfunc.round_down_to_odd(smooth_window_len_pixels)
    # Use your actual mathfunc or a direct odd check:
    window_len = int(smooth_window_len_pixels)
    if window_len % 2 == 0: window_len +=1
    window_len = max(3, window_len)


    if len(jagged_envelope) < window_len :
        # print(f"Warning: Jagged envelope length ({len(jagged_envelope)}) < smooth window ({window_len}). Returning unsmoothed.")
        return jagged_envelope # Not enough points to smooth properly

    smoothed_envelope = _smooth(jagged_envelope, window_len=window_len, window=smooth_window_type, mode="same")
    
    return smoothed_envelope
from scipy.signal import savgol_filter # Add this import

def fit_envelope_savgol(
    extrema_x, extrema_y,
    full_x_axis, # Envelope will be defined on this grid
    savgol_window_len_pixels=51,
    savgol_polyorder=3,
    verbose=False
):
    extrema_x = np.asarray(extrema_x); extrema_y = np.asarray(extrema_y)
    full_x_axis = np.asarray(full_x_axis)

    if len(extrema_x) < 2: # Need at least two points for robust interpolation
        if verbose: print(f"  Warning: Not enough extrema ({len(extrema_x)}) to create Savgol envelope. Returning flat line or NaNs.")
        return np.full_like(full_x_axis, np.median(extrema_y) if len(extrema_y) > 0 else np.nan, dtype=float)

    # Ensure extrema_x are sorted for np.interp
    sort_indices = np.argsort(extrema_x)
    sorted_extrema_x = extrema_x[sort_indices]; sorted_extrema_y = extrema_y[sort_indices]
    
    # 1. Linear Interpolation to get a jagged envelope on the full_x_axis grid
    jagged_envelope = np.interp(
        full_x_axis, sorted_extrema_x, sorted_extrema_y,
        left=sorted_extrema_y[0], right=sorted_extrema_y[-1]
    )

    # 2. Smooth with Savitzky-Golay
    window_len = mathfunc.round_down_to_odd(int(savgol_window_len_pixels))
    # Savgol window must be odd and greater than polyorder
    if window_len <= savgol_polyorder:
        window_len = savgol_polyorder + 1 if savgol_polyorder % 2 == 0 else savgol_polyorder + 2
        window_len = mathfunc.round_down_to_odd(window_len) # Ensure odd
    window_len = max(3, window_len)


    if len(jagged_envelope) < window_len :
        if verbose: print(f"  Warning: Jagged envelope length ({len(jagged_envelope)}) < Savgol window ({window_len}). Returning unsmoothed jagged envelope.")
        return jagged_envelope

    try:
        # mode="interp" can sometimes be better for edges if data is not periodic
        # mode="mirror" is also a common choice
        smoothed_envelope = savgol_filter(jagged_envelope, window_length=window_len, polyorder=savgol_polyorder, mode="mirror")
    except ValueError as e:
        if verbose: print(f"  Warning: Savgol filter failed ({e}). Returning jagged envelope.")
        return jagged_envelope 
        
    return smoothed_envelope

def fit_envelope_polynomial(x_coords, y_coords, degree=3, robust=True):
    x_coords_arr = np.asarray(x_coords).reshape(-1, 1) # Ensure 2D for scaler
    y_coords_arr = np.asarray(y_coords)

    if len(x_coords_arr) < degree + 1:
        print(f"Warning: Not enough points ({len(x_coords_arr)}) to fit envelope of degree {degree}. Returning None.")
        return None, None, None # Model, PolyFeatures, Scaler

    # --- Normalize x_coords ---
    # MinMaxScaler to [0, 1] or [-1, 1] is common
    # StandardScaler (mean 0, std 1) also works well
    # Let's use MinMaxScaler to [-1, 1] for better behavior around origin if poly has offset
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaled = x_scaler.fit_transform(x_coords_arr)
    # --- End Normalization ---

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    # Fit PolynomialFeatures on SCALED x data
    X_poly_scaled = poly_features.fit_transform(x_scaled)

    if robust:
        model = HuberRegressor(epsilon=1.35, max_iter=500) # Increase max_iter
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(fit_intercept=False) # PolynomialFeatures includes bias

    try:
        model.fit(X_poly_scaled, y_coords_arr)
        return model, poly_features, x_scaler # Return the scaler
    except ValueError as e:
        print(f"Error fitting envelope polynomial: {e}")
        return None, None, None
    except Exception as e_gen: # Catch other potential convergence issues
        print(f"General error/convergence failure fitting envelope: {e_gen}")
        return None, None, None


def predict_envelope(x_coords_new, model, poly_features_transformer, x_scaler):
    """Predicts envelope y-values for new x-coordinates, using the trained scaler."""
    if model is None or poly_features_transformer is None or x_scaler is None:
        return np.full_like(np.asarray(x_coords_new), np.nan)

    x_coords_new_arr = np.asarray(x_coords_new).reshape(-1, 1)
    # --- Scale new x_coords using the *same* scaler from fitting ---
    x_new_scaled = x_scaler.transform(x_coords_new_arr)
    # --- End Scaling ---

    X_new_poly_scaled = poly_features_transformer.transform(x_new_scaled)
    return model.predict(X_new_poly_scaled)
    
def predict_spacing(x_midpoints_new, spacing_model, spacing_poly_transformer):
    """Predicts expected spacing for new x_midpoints."""
    if spacing_model is None or spacing_poly_transformer is None:
        return np.full_like(np.asarray(x_midpoints_new), np.nan) # Default if no model
    X_new_poly = spacing_poly_transformer.transform(np.asarray(x_midpoints_new).reshape(-1, 1))
    return spacing_model.predict(X_new_poly)


def remove_false_triplets(
    all_extrema_x, all_extrema_y, all_extrema_types,
    max_triplet_x_span_pixels, # Max x-distance for a P-V-P or V-P-V to be a "triplet"
                               # e.g., 0.3 * typical_LFC_period
    min_prominence_depth_factor_for_center, # Center extremum's prom/depth must be at least
                                            # this factor of its immediate neighbors' difference
                                            # to be considered "real" within a triplet. e.g., 0.1
    verbose=False
):
    """
    Identifies and removes spurious triplets (P-V-P or V-P-V) with small x-span
    and a weak central extremum.

    Args:
        all_extrema_x, all_extrema_y, all_extrema_types: Combined, sorted, and alternated list.
        max_triplet_x_span_pixels: Maximum x-width for a sequence of 3 to be a triplet.
        min_prominence_depth_factor_for_center: How significant the central extremum
                                                must be relative to its direct neighbors.
    Returns:
        tuple: (filtered_x, filtered_y, filtered_types)
    """
    if len(all_extrema_x) < 3:
        return all_extrema_x, all_extrema_y, all_extrema_types

    filtered_x, filtered_y, filtered_types = [], [], []
    
    # Convert to lists for easier manipulation (pop, insert) if needed, or use boolean mask
    current_x = list(all_extrema_x)
    current_y = list(all_extrema_y)
    current_types = list(all_extrema_types)
    
    indices_to_keep = np.ones(len(current_x), dtype=bool)
    
    i = 0
    fig = plt.figure()
    ax = fig.add_subplot()
    while i < len(current_x) - 3:
        # Check for P-V-P or V-P-V triplet
        x1, y1, t1 = current_x[i], current_y[i], current_types[i]
        x2, y2, t2 = current_x[i+1], current_y[i+1], current_types[i+1]
        x3, y3, t3 = current_x[i+2], current_y[i+2], current_types[i+2]
        xm2, ym2, tm2 = current_x[i], current_y[i], current_types[i]

        # Ensure it's an alternating triplet (t1 != t2 and t2 != t3 implies t1 == t3)
        if t1 == t3 and t1 != t2:
            triplet_x_span = x3 - x1
            if triplet_x_span < max_triplet_x_span_pixels:
                # Potential false triplet due to small x-span
                # Now check the significance of the central extremum (E2)
                # E.g., for P1-V2-P3, check depth of V2 relative to P1, P3
                # For V1-P2-V3, check prominence of P2 relative to V1, V3
                [ax.scatter(x_,y_,c='k',marker='.') for (x_,y_) in [(x1,y1),(x2,y2),(x3,y3)]]
                if t1 == 1: # Triplet is Peak(y1) - Valley(y2) - Peak(y3)
                    # Check depth of central valley y2
                    depth1 = y1 - y2 # Depth relative to left peak
                    depth2 = y3 - y2 # Depth relative to right peak
                    # Significance: is the valley (y2) substantially lower than its peaks?
                    # Or, is the valley's depth a small fraction of the peak heights?
                    # Let's use: depth must be a certain fraction of the smaller peak's height from valley
                    # max_peak_height_diff = max(y1,y3) - min(y1,y3) # Not quite right
                    # Instead, check prominence of P1 over V2, and P3 over V2
                    # No, check depth of V2 relative to an interpolated line between P1 and P3
                    # Or simpler: depth of V2 relative to min(P1,P3)
                    effective_depth = min(depth1, depth2)
                    reference_amplitude = max(abs(y1), abs(y3)) # Or abs(y1-y3)
                    # A more direct check: is y2 "sandwiched" shallowly?
                    # The central valley's depth should be significant compared to the height difference of its bracketing peaks
                    height_diff_peaks = abs(y1 - y3)
                    
                    # If the valley y2 is not significantly lower than min(y1, y3)
                    # A simple prominence check for the central valley y2
                    # Prominence from left: y1 - y2. Prominence from right: y3 - y2
                    # Check if y2 is "significant" relative to the y-range defined by y1 and y3.
                    # Prominence of the central valley:
                    prominence_of_valley = min(y1 - y2, y3 - y2) # How much it dips below the lower peak
                    # Reference amplitude for normalization: span of the outer peaks
                    amplitude_span_outer = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3))
                    
                    if prominence_of_valley < min_prominence_depth_factor_for_center * amplitude_span_outer \
                       or prominence_of_valley < 0.01 * min(abs(y1), abs(y3)): # Or some absolute small value related to noise
                        if verbose: print(f"  Triplet P-V-P at x~{x2:.1f} (span {triplet_x_span:.1f}px): Valley y2={y2:.2f} too shallow "
                                          f"(prom={prominence_of_valley:.2f}) relative to P1={y1:.2f}, P3={y3:.2f}. Removing V2.")
                        indices_to_keep[i+1] = False # Remove central valley
                        plt.scatter(x2,y2,marker='x',c='r')
                        # Potentially remove one of the peaks if they become too close
                        # For now, just remove center. The next _ensure_alternation pass will handle P-P.
                        i += 1 # Skip checking the next element as part of a new triplet start
                               # because we modified the structure (or effectively will)
                        continue 

                elif t1 == -1: # Triplet is Valley(y1) - Peak(y2) - Valley(y3)
                    # Check prominence of central peak y2
                    prom1 = y2 - y1 # Prominence relative to left valley
                    prom2 = y2 - y3 # Prominence relative to right valley
                    # Significance: is the peak y2 substantially higher than its valleys?
                    effective_prominence = min(prom1, prom2)
                    # Reference amplitude for normalization: span of the outer valleys
                    amplitude_span_outer = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3))

                    if effective_prominence < min_prominence_depth_factor_for_center * amplitude_span_outer \
                       or effective_prominence < 0.1 * min(abs(y1), abs(y3)): # Heuristic: if prominence too small
                        if verbose: print(f"  Triplet V-P-V at x~{x2:.1f} (span {triplet_x_span:.1f}px): Peak y2={y2:.2f} too weak "
                                          f"(prom={effective_prominence:.2f}) relative to V1={y1:.2f}, V3={y3:.2f}. Removing P2.")
                        indices_to_keep[i+1] = False # Remove central peak
                        i += 1 
                        continue
        i += 1
        
    plt.show()
    filtered_x = all_extrema_x[indices_to_keep]
    filtered_y = all_extrema_y[indices_to_keep]
    filtered_types = all_extrema_types[indices_to_keep]
    
    # After removing points, re-alternate to fix any P-P or V-V sequences
    # Use the standard _ensure_alternation
    temp_peaks_x = filtered_x[filtered_types == 1]
    temp_peaks_y = filtered_y[filtered_types == 1]
    temp_valleys_x = filtered_x[filtered_types == -1]
    temp_valleys_y = filtered_y[filtered_types == -1]

    final_peaks_x, final_peaks_y, \
    final_valleys_x, final_valleys_y = _ensure_alternation(
        (temp_peaks_x, temp_peaks_y), (temp_valleys_x, temp_valleys_y)
    )
    
    # Recombine into combined lists for return consistency if needed by downstream
    # For now, just return separated and let the caller combine if needed
    # Or, more robustly, return combined lists from this function too
    
    # To return combined lists similar to _ensure_alternation_and_tag:
    if final_peaks_x.size == 0 and final_valleys_x.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]) # Keep original return for separated
        # For combined: return np.array([]), np.array([]), np.array([])
        
    all_final_x = np.concatenate((final_peaks_x, final_valleys_x)) if final_peaks_x.size > 0 or final_valleys_x.size > 0 else np.array([])
    all_final_y = np.concatenate((final_peaks_y, final_valleys_y)) if final_peaks_y.size > 0 or final_valleys_y.size > 0 else np.array([])
    all_final_types = np.concatenate((np.ones(final_peaks_x.size), -np.ones(final_valleys_x.size))) if all_final_x.size > 0 else np.array([])

    if all_final_x.size > 0:
        sort_idx = np.argsort(all_final_x)
        return final_peaks_x, final_peaks_y, final_valleys_x, final_valleys_y, \
               all_final_x[sort_idx], all_final_y[sort_idx], all_final_types[sort_idx]
    else: # No points left
        return np.array([]), np.array([]), np.array([]), np.array([]), \
               np.array([]), np.array([]), np.array([])
               
               


def remove_false_triplets_v2( # Renamed for clarity
    all_extrema_x_in, all_extrema_y_in, all_extrema_types_in, # Combined, sorted, alternated
    max_triplet_x_span_pixels,
    min_prom_depth_factor_center, # For check against immediate opposites
    y_consistency_factor=0.2,     # For new check: y_center vs (y_prev_same, y_next_same)
                                  # y_center must be within [min - factor*diff, max + factor*diff]
    verbose=False
):
    """
    Identifies and removes spurious triplets (P-V-P or V-P-V).
    A triplet's central extremum is removed if:
    1. Its x-span is small AND
    2. Its prominence/depth relative to immediate opposites is small OR
    3. Its y-value is inconsistent with its further same-type neighbors.
    """
    if len(all_extrema_x_in) < 3:
        return all_extrema_x_in, all_extrema_y_in, all_extrema_types_in # No change, return as is

    # Work on copies
    current_x = list(all_extrema_x_in)
    current_y = list(all_extrema_y_in)
    current_types = list(all_extrema_types_in)
    
    indices_to_remove = [] # Store indices of central elements of false triplets
    
    # fig = plt.figure()
    # ax = fig.add_subplot()
    
    # Iterate up to the point where a triplet can still be formed
    for i in range(len(current_x) - 2):
        x1, y1, t1 = current_x[i], current_y[i], current_types[i]
        x2, y2, t2 = current_x[i+1], current_y[i+1], current_types[i+1]
        x3, y3, t3 = current_x[i+2], current_y[i+2], current_types[i+2]

        # Check for an alternating triplet (P-V-P or V-P-V)
        # This should always be true if input is already alternated, but good check.
        if not (t1 == t3 and t1 != t2):
            continue 

        triplet_x_span = x3 - x1
        is_potential_false_triplet = False

        if triplet_x_span < max_triplet_x_span_pixels:
            # --- Check 1: Prominence/Depth of central E2 relative to E1, E3 ---
            center_is_weak = False
            if t1 == 1: # P1-V2-P3, check depth of V2
                prominence_of_valley = min(y1 - y2, y3 - y2)
                amplitude_span_outer_peaks = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if prominence_of_valley < min_prom_depth_factor_center * amplitude_span_outer_peaks or prominence_of_valley <= 0:
                    center_is_weak = True
                    if verbose: print(f"  Triplet P-V-P at x~{x2:.1f} (span {triplet_x_span:.1f}): Center V2 (y={y2:.2f}) weak prom={prominence_of_valley:.2f} rel to P1={y1:.2f}, P3={y3:.2f}.")
            else: # V1-P2-V3, check prominence of P2
                effective_prominence = min(y2 - y1, y2 - y3)
                amplitude_span_outer_valleys = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if effective_prominence < min_prom_depth_factor_center * amplitude_span_outer_valleys or effective_prominence <=0:
                    center_is_weak = True
                    if verbose: print(f"  Triplet V-P-V at x~{x2:.1f} (span {triplet_x_span:.1f}): Center P2 (y={y2:.2f}) weak prom={effective_prominence:.2f} rel to V1={y1:.2f}, V3={y3:.2f}.")

            # --- Check 2: Y-value consistency of E2 with further same-type neighbors ---
            center_y_inconsistent = False
            if not center_is_weak: # Only do this check if it wasn't already flagged as weak
                # Find E0 (same type as E2, before E1) and E4 (same type as E2, after E3)
                # This requires searching the *original unfiltered list* before any removals in this loop
                
                # Find previous same-type extremum (E0)
                e0_idx = -1
                for k in range(i - 1, -1, -1): # Search backwards from E1
                    if all_extrema_types_in[k] == t2: # t2 is type of central element E2
                        e0_idx = k
                        break
                
                # Find next same-type extremum (E4)
                e4_idx = -1
                for k in range(i + 3, len(all_extrema_x_in)): # Search forwards from E3
                    if all_extrema_types_in[k] == t2:
                        e4_idx = k
                        break
                
                if e0_idx != -1 and e4_idx != -1: # Both E0 and E4 exist
                    y0 = all_extrema_y_in[e0_idx]
                    y4 = all_extrema_y_in[e4_idx]
                    
                    min_y_outer_same = min(y0, y4)
                    max_y_outer_same = max(y0, y4)
                    y_diff_outer_same = abs(y0 - y4)
                    
                    lower_bound = min_y_outer_same - y_consistency_factor * y_diff_outer_same
                    upper_bound = max_y_outer_same + y_consistency_factor * y_diff_outer_same
                    
                    if not (lower_bound <= y2 <= upper_bound):
                        center_y_inconsistent = True
                        if verbose: print(f"  Triplet ...E2... at x~{x2:.1f}: Center E2 (y={y2:.2f}, type={t2}) y-inconsistent "
                                          f"with same-type neighbors E0(y={y0:.2f}), E4(y={y4:.2f}). Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                # else: Not enough same-type neighbors to perform this check, so it passes this sub-check.

            if center_is_weak or center_y_inconsistent:
                is_potential_false_triplet = True

            if is_potential_false_triplet:
                # Mark the central element for removal
                # We use original indices from all_extrema_x_in to mark on a boolean mask
                # Find the index in all_extrema_x_in that corresponds to current_x[i+1]
                # This is tricky if current_x has already been modified.
                # Better to operate on a boolean mask of the original input.
                indices_to_remove.append(i+1) # This index is for the 'current_x' list *as it's being processed*
                                              # This needs to be carefully handled if list is modified in place.
                                              # Using a boolean mask on original is safer.

    # --- Safer removal using a boolean mask on the input arrays ---
    if len(all_extrema_x_in) == 0 : # Should have been caught earlier
        return np.array([]), np.array([]), np.array([]), np.array([]) \
               ,np.array([]), np.array([]), np.array([])


    indices_to_keep_mask = np.ones(len(all_extrema_x_in), dtype=bool)
    
    # Re-iterate for removal logic, this time operating on the original input via mask
    # This is safer than modifying the list being iterated over.
    current_x_orig = all_extrema_x_in
    current_y_orig = all_extrema_y_in
    current_types_orig = all_extrema_types_in

    processed_central_indices = set() # To avoid re-evaluating a removed center

    for i in range(len(current_x_orig) - 2):
        if i+1 in processed_central_indices:
            continue

        x1, y1, t1 = current_x_orig[i], current_y_orig[i], current_types_orig[i]
        x2, y2, t2 = current_x_orig[i+1], current_y_orig[i+1], current_types_orig[i+1]
        x3, y3, t3 = current_x_orig[i+2], current_y_orig[i+2], current_types_orig[i+2]

        if not (t1 == t3 and t1 != t2):
            continue

        triplet_x_span = x3 - x1
        remove_center = False

        if triplet_x_span < max_triplet_x_span_pixels:
            center_is_weak = False
            if t1 == 1: # P1-V2-P3
                prom_of_V2 = min(y1 - y2, y3 - y2)
                amp_span_P = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if prom_of_V2 < min_prom_depth_factor_center * amp_span_P or prom_of_V2 <= 0:
                    center_is_weak = True
            else: # V1-P2-V3
                eff_prom_P2 = min(y2 - y1, y2 - y3)
                amp_span_V = abs(y1 - y3) if abs(y1-y3) > 1e-6 else max(abs(y1),abs(y3), 1e-6)
                if eff_prom_P2 < min_prom_depth_factor_center * amp_span_V or eff_prom_P2 <=0:
                    center_is_weak = True
            
            if center_is_weak:
                remove_center = True
                if verbose: print(f"  Triplet at x~{x2:.1f}: Center E2 (y={y2:.2f}) weak prominence/depth. Marked for removal.")
            else: # Center not weak by first criterion, check y-consistency
                e0_idx, e4_idx = -1, -1
                # Search in original full list (all_extrema_x_in) for same-type neighbors
                # This search needs to be based on the original indices, not 'i'
                # Find original index of current_x_orig[i]
                original_idx_of_e1 = np.where(all_extrema_x_in == x1)[0] # Assuming x are unique enough
                original_idx_of_e3 = np.where(all_extrema_x_in == x3)[0]
                if not (original_idx_of_e1.size > 0 and original_idx_of_e3.size > 0): continue # Should not happen

                for k in range(original_idx_of_e1[0] - 1, -1, -1):
                    if all_extrema_types_in[k] == t2: e0_idx = k; break
                for k in range(original_idx_of_e3[0] + 1, len(all_extrema_x_in)):
                    if all_extrema_types_in[k] == t2: e4_idx = k; break
                
                if e0_idx != -1 and e4_idx != -1:
                    y0, y4 = all_extrema_y_in[e0_idx], all_extrema_y_in[e4_idx]
                    min_y_outer, max_y_outer = min(y0,y4), max(y0,y4)
                    y_diff_outer = abs(y0-y4)
                    lower_b = min_y_outer - y_consistency_factor * y_diff_outer
                    upper_b = max_y_outer + y_consistency_factor * y_diff_outer
                    if not (lower_b <= y2 <= upper_b):
                        remove_center = True
                        if verbose: print(f"  Triplet at x~{x2:.1f}: Center E2 (y={y2:.2f}) y-inconsistent. Marked for removal.")
            
            if remove_center:
                # [ax.scatter(x_,y_,c='k',marker='.') for (x_,y_) in [(x1,y1),(x2,y2),(x3,y3)]]
                # ax.scatter(x2,y2,marker='x',c='r')
                indices_to_keep_mask[np.where(all_extrema_x_in == x2)[0][0]] = False # Mark central for removal
                processed_central_indices.add(i+1) # Mark this original central index as processed

    filtered_x_final = all_extrema_x_in[indices_to_keep_mask]
    filtered_y_final = all_extrema_y_in[indices_to_keep_mask]
    filtered_types_final = all_extrema_types_in[indices_to_keep_mask]

    # Final alternation pass
    # The _ensure_alternation_and_tag also separates, so we can use its separated output
    peaks_x_out, peaks_y_out, valleys_x_out, valleys_y_out, \
    comb_x_out, comb_y_out, comb_types_out = _ensure_alternation_and_tag(
        filtered_x_final[filtered_types_final==1], filtered_y_final[filtered_types_final==1],
        filtered_x_final[filtered_types_final==-1], filtered_y_final[filtered_types_final==-1]
    )
    
    return peaks_x_out, peaks_y_out, valleys_x_out, valleys_y_out, \
           comb_x_out, comb_y_out, comb_types_out
           
           
def filter_lfc_extrema_v5(
    peaks_x_cand, peaks_y_cand_refined,
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis,
    # Parameters
    spacing_poly_degree=2,
    spacing_uncertainty_const=2.0,
    spacing_max_dev_factor=3.0,
    envelope_poly_degree=3,
    max_allowed_Venv_Penv_ratio = 0.3, # Single threshold for V_env/P_env
    snr_min_thresh_poisson=5.0,
    noise_estimation_window_pixels=15, # For robust_local_noise_std
    global_fallback_noise_std=None,
    plot_filter_diagnostics=False,
    verbose=False
):
    # --- 0. Initial Alternation & Setup ---
    if global_fallback_noise_std is None:
        global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))

    current_peaks_x, current_peaks_y, \
    current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined),
        (valleys_x_cand, valleys_y_cand_refined)
    )
    if verbose: print(f"  RuleFilter v5_simp_ratio Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")

    # Store initial alternated candidates for plotting reference
    plot_cand_peaks_x_alt = np.copy(current_peaks_x)
    plot_cand_peaks_y_alt = np.copy(current_peaks_y)
    plot_cand_valleys_x_alt = np.copy(current_valleys_x)
    plot_cand_valleys_y_alt = np.copy(current_valleys_y)

    if not current_peaks_x.size or not current_valleys_x.size:
        print("  Not enough peaks OR valleys after initial alternation for filtering.")
        plot_data_placeholder = (None,None,None, np.array([]),np.array([]),np.array([])) # (unscaled, scaled, labels, cand_x, cand_y, cand_types)
        return current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y, plot_data_placeholder

    # --- 1. Fit Global Spacing Models ---
    spacing_model_p, spacing_poly_trans_p = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v, spacing_poly_trans_v = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)

    # --- 2. Fit Global Envelope Models (with X normalization) ---
    envelope_model_p, envelope_poly_trans_p, x_scaler_p = None, None, None
    if current_peaks_x.size > envelope_poly_degree :
        envelope_model_p, envelope_poly_trans_p, x_scaler_p = fit_envelope_polynomial(current_peaks_x, current_peaks_y, degree=envelope_poly_degree)
    if envelope_model_p is None and verbose: print("  Warning: Failed to fit peak envelope model.")
    
    envelope_model_v, envelope_poly_trans_v, x_scaler_v = None, None, None
    if current_valleys_x.size > envelope_poly_degree:
        envelope_model_v, envelope_poly_trans_v, x_scaler_v = fit_envelope_polynomial(current_valleys_x, current_valleys_y, degree=envelope_poly_degree)
    if envelope_model_v is None and verbose: print("  Warning: Failed to fit valley envelope model.")

    # --- Data structures for metrics ---
    peak_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all':[]}
    valley_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all':[]}

    kept_peaks_x, kept_peaks_y = [], []
    # --- Filter Peaks ---
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        peak_metrics['x'].append(xp); peak_metrics['y'].append(yp)
        pass_snr_p, pass_spacing_p, pass_ratio_p = True, True, True

        # S/N Calculation
        left_v_idx=np.where(current_valleys_x<xp)[0]; right_v_idx=np.where(current_valleys_x>xp)[0]
        baseline_snr_P=np.nan
        if left_v_idx.size>0 and right_v_idx.size>0: 
            baseline_snr_P=max(current_valleys_y[left_v_idx[-1]],current_valleys_y[right_v_idx[0]])
        elif left_v_idx.size>0: 
            baseline_snr_P=current_valleys_y[left_v_idx[-1]]
        elif right_v_idx.size>0: 
            baseline_snr_P=current_valleys_y[right_v_idx[0]]
        signal_P = yp - baseline_snr_P if np.isfinite(baseline_snr_P) else yp - np.median(original_y_axis)
        signal_P = max(0, signal_P)
        noise_approx_P = np.sqrt(max(1, yp) + max(1, baseline_snr_P if np.isfinite(baseline_snr_P) else np.median(original_y_axis)))
        current_snr_P = signal_P / noise_approx_P if noise_approx_P > 1e-9 else signal_P / 1e-9
        peak_metrics['snr'].append(current_snr_P)
        if current_snr_P < snr_min_thresh_poisson: pass_snr_p = False

        # Spacing Consistency
        current_spacing_dev_p = np.nan
        if spacing_model_p and pass_snr_p: # Only check if S/N passed
            devs_norm_list = [] # ... (spacing logic as before) ...
            if i > 0:
                s_bwd_act = xp-current_peaks_x[i-1]; x_mid_bwd=(xp+current_peaks_x[i-1])/2.0
                s_bwd_exp_arr = predict_spacing(x_mid_bwd, spacing_model_p, spacing_poly_trans_p)
                if s_bwd_exp_arr is not None and np.isfinite(s_bwd_exp_arr[0]): devs_norm_list.append(abs(s_bwd_act-s_bwd_exp_arr[0])/spacing_uncertainty_const)
            if i < len(current_peaks_x)-1:
                s_fwd_act = current_peaks_x[i+1]-xp; x_mid_fwd=(xp+current_peaks_x[i+1])/2.0
                s_fwd_exp_arr = predict_spacing(x_mid_fwd, spacing_model_p, spacing_poly_trans_p)
                if s_fwd_exp_arr is not None and np.isfinite(s_fwd_exp_arr[0]): devs_norm_list.append(abs(s_fwd_act-s_fwd_exp_arr[0])/spacing_uncertainty_const)
            if devs_norm_list: current_spacing_dev_p=np.mean(devs_norm_list)
            else: current_spacing_dev_p=0 
            if current_spacing_dev_p > spacing_max_dev_factor: pass_spacing_p=False
        peak_metrics['spacing_dev'].append(current_spacing_dev_p)

        # Envelope Ratio (Simplified)
        obs_r_peak = np.nan
        if pass_snr_p and pass_spacing_p: # Only check if previous passed
            if envelope_model_p and envelope_model_v and x_scaler_p and x_scaler_v:
                P_env_arr = predict_envelope(xp, envelope_model_p, envelope_poly_trans_p, x_scaler_p)
                V_env_arr = predict_envelope(xp, envelope_model_v, envelope_poly_trans_v, x_scaler_v)
                if P_env_arr is not None and V_env_arr is not None and np.isfinite(P_env_arr[0]) and np.isfinite(V_env_arr[0]):
                    P_env_val, V_env_val = P_env_arr[0], V_env_arr[0]
                    if yp <= V_env_val: pass_ratio_p = False # Peak must be above valley envelope
                    if pass_ratio_p and P_env_val > 1e-6 and P_env_val > V_env_val: # Meaningful P_env and modulation
                        obs_r_peak = V_env_val / P_env_val
                        if obs_r_peak > max_allowed_Venv_Penv_ratio: pass_ratio_p = False
                    elif pass_ratio_p and P_env_val <= V_env_val : # No modulation or P_env too low
                        pass_ratio_p = False # Fail if no clear peak dominance by envelopes
                else: pass_ratio_p = False # Envelope prediction failed
            elif envelope_model_p is None or envelope_model_v is None: # Models not available
                pass_ratio_p = True # Cannot test, so pass this criterion
        peak_metrics['env_ratio_obs'].append(obs_r_peak)
        # peak_metrics['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio) # Store threshold for plotting

        peak_metrics['pass_snr'].append(pass_snr_p); peak_metrics['pass_spacing'].append(pass_spacing_p); peak_metrics['pass_ratio'].append(pass_ratio_p)
        if pass_snr_p and pass_spacing_p and pass_ratio_p:
            kept_peaks_x.append(xp); kept_peaks_y.append(yp)
            peak_metrics['pass_all'].append(True)
        else:
            peak_metrics['pass_all'].append(False)


    # --- Filter Valleys (Similar logic) ---
    kept_valleys_x, kept_valleys_y = [], []
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        valley_metrics['x'].append(xv); valley_metrics['y'].append(yv)
        pass_snr_v, pass_spacing_v, pass_ratio_v = True, True, True

        # S/N
        # ... (Valley S/N logic as before) ...
        left_p_idx=np.where(current_peaks_x<xv)[0]
        right_p_idx=np.where(current_peaks_x>xv)[0]
        baseline_snr_V=np.nan
        if left_p_idx.size>0 and right_p_idx.size>0:
            baseline_snr_V=min(current_peaks_y[left_p_idx[-1]],current_peaks_y[right_p_idx[0]])
        elif left_p_idx.size>0:
            baseline_snr_V=current_peaks_y[left_p_idx[-1]]
        elif right_p_idx.size>0:
            baseline_snr_V=current_peaks_y[right_p_idx[0]]
        signal_V=baseline_snr_V-yv if np.isfinite(baseline_snr_V) else np.median(original_y_axis)-yv;signal_V=max(0,signal_V)
        noise_approx_V=np.sqrt(max(1,abs(yv))+max(1,baseline_snr_V if np.isfinite(baseline_snr_V) else np.median(original_y_axis)))
        current_snr_V=signal_V/noise_approx_V if noise_approx_V>1e-9 else signal_V/1e-9
        valley_metrics['snr'].append(current_snr_V)
        if current_snr_V < snr_min_thresh_poisson: pass_snr_v = False

        # Spacing
        current_spacing_dev_v = np.nan
        if spacing_model_v and pass_snr_v:
            # ... (Valley spacing logic as before) ...
            devs_v=[]
            if i > 0:
                s_b,xm_b=xv-current_valleys_x[i-1],(xv+current_valleys_x[i-1])/2
                s_e_arr=predict_spacing(xm_b,spacing_model_v,spacing_poly_trans_v)
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):devs_v.append(abs(s_b-s_e_arr[0])/spacing_uncertainty_const)
            if i < len(current_valleys_x)-1:
                s_f,xm_f=current_valleys_x[i+1]-xv,(xv+current_valleys_x[i+1])/2
                s_e_arr=predict_spacing(xm_f,spacing_model_v,spacing_poly_trans_v)
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):devs_v.append(abs(s_f-s_e_arr[0])/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v=np.mean(devs_v)
            else:current_spacing_dev_v=0
            if current_spacing_dev_v > spacing_max_dev_factor: pass_spacing_v=False
        valley_metrics['spacing_dev'].append(current_spacing_dev_v)
        
        # Envelope Ratio (Simplified)
        obs_r_valley = np.nan
        if pass_snr_v and pass_spacing_v:
            if envelope_model_p and envelope_model_v and x_scaler_p and x_scaler_v:
                P_env_arr = predict_envelope(xv, envelope_model_p, envelope_poly_trans_p, x_scaler_p)
                V_env_arr = predict_envelope(xv, envelope_model_v, envelope_poly_trans_v, x_scaler_v)
                if P_env_arr is not None and V_env_arr is not None and np.isfinite(P_env_arr[0]) and np.isfinite(V_env_arr[0]):
                    P_env_val, V_env_val = P_env_arr[0], V_env_arr[0]
                    if yv >= P_env_val: pass_ratio_v = False # Valley must be below peak envelope
                    if pass_ratio_v and P_env_val > 1e-6 and P_env_val > V_env_val: # Meaningful P_env and modulation
                        obs_r_valley = V_env_val / P_env_val
                        if obs_r_valley > max_allowed_Venv_Penv_ratio: pass_ratio_v = False
                    elif pass_ratio_v and P_env_val <= V_env_val: # No modulation by envelopes
                        pass_ratio_v = False
                else: pass_ratio_v = False
            elif not envelope_model_p or not envelope_model_v: pass_ratio_v = True # Cannot test
        valley_metrics['env_ratio_obs'].append(obs_r_valley)
        # valley_metrics['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio)

        valley_metrics['pass_snr'].append(pass_snr_v); valley_metrics['pass_spacing'].append(pass_spacing_v); valley_metrics['pass_ratio'].append(pass_ratio_v)
        if pass_snr_v and pass_spacing_v and pass_ratio_v:
            kept_valleys_x.append(xv); kept_valleys_y.append(yv)
            valley_metrics['pass_all'].append(True)
        else:
            valley_metrics['pass_all'].append(False)

    # Convert metric lists to numpy arrays
    for key in peak_metrics: peak_metrics[key] = np.array(peak_metrics[key])
    for key in valley_metrics: valley_metrics[key] = np.array(valley_metrics[key])

    # --- Plotting Diagnostics ---
    if plot_filter_diagnostics:
        # ... (Plotting logic from previous response, using updated metrics) ...
        # ... (Panel 4 for Envelope Ratio will now plot peak_metrics['env_ratio_obs']
        #      and valley_metrics['env_ratio_obs'] against a horizontal line at
        #      max_allowed_Venv_Penv_ratio_threshold)
        fig, axs = plt.subplots(4, 1, figsize=(15,16), sharex=True); fig.suptitle("Rule-Based Filter Diagnostics (Simplified Ratio)", fontsize=16)
        xpr = np.linspace(np.min(original_x_axis),np.max(original_x_axis),300) if original_x_axis.size>0 else np.array([])
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig Data")
        if envelope_model_p and x_scaler_p and xpr.size>0: 
            axs[0].plot(xpr,
                        predict_envelope(xpr,envelope_model_p,envelope_poly_trans_p,x_scaler_p),
                        'b-',alpha=0.6,lw=1.5,label="P Env")
        if envelope_model_v and x_scaler_v and xpr.size>0: 
            axs[0].plot(xpr,
                        predict_envelope(xpr,envelope_model_v,envelope_poly_trans_v,x_scaler_v),
                        'g-',alpha=0.6,lw=1.5,label="V Env")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',
                       marker='^',s=20,alpha=0.5,label="P Cands");
        axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,
                       c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if peak_metrics['x'][peak_metrics['pass_all']].size>0: 
            axs[0].scatter(peak_metrics['x'][peak_metrics['pass_all']],
                           peak_metrics['y'][peak_metrics['pass_all']],
                           c='blue',marker='^',s=40,label="Kept P")
        if valley_metrics['x'][valley_metrics['pass_all']].size>0: 
            axs[0].scatter(valley_metrics['x'][valley_metrics['pass_all']],
                           valley_metrics['y'][valley_metrics['pass_all']],
                           c='darkgreen',marker='v',s=40,label="Kept V")
        axs[0].set_title("Data, Envelopes, Filtered Extrema")
        axs[0].set_ylabel("Flux")
        axs[0].legend(fontsize='small',loc='upper right')
        axs[0].grid(True,ls=':')
        
        # S/N Panel
        if peak_metrics['x'].size>0:
            axs[1].scatter(peak_metrics['x'][peak_metrics['pass_snr']],
                           peak_metrics['snr'][peak_metrics['pass_snr']],
                           c='blue',marker='^',s=30,alpha=0.7,label="Pass S/N(P)")
            axs[1].scatter(peak_metrics['x'][~peak_metrics['pass_snr']],
                           peak_metrics['snr'][~peak_metrics['pass_snr']],
                           c='red',marker='x',s=50,label="Fail S/N(P)")
        if valley_metrics['x'].size>0:
            axs[1].scatter(valley_metrics['x'][valley_metrics['pass_snr']],
                           valley_metrics['snr'][valley_metrics['pass_snr']],
                           c='green',marker='v',s=30,alpha=0.7,label="Pass S/N(V)");
            axs[1].scatter(valley_metrics['x'][~valley_metrics['pass_snr']],
                           valley_metrics['snr'][~valley_metrics['pass_snr']],
                           c='magenta',marker='x',s=50,label="Fail S/N(V)")
        axs[1].axhline(snr_min_thresh_poisson,c='k',ls='--',label=f"S/N Thresh")
        axs[1].set_title("S/N")
        axs[1].set_ylabel("S/N Val")
        axs[1].legend(fontsize='small')
        axs[1].grid(True,ls=':')
        # axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        
        # Spacing Panel
        if peak_metrics['x'].size>0:
            axs[2].scatter(peak_metrics['x'][peak_metrics['pass_spacing']],
                           peak_metrics['spacing_dev'][peak_metrics['pass_spacing']],
                           c='blue',marker='^',s=30,alpha=0.7,label="Pass Spacing(P)");
            axs[2].scatter(peak_metrics['x'][~peak_metrics['pass_spacing']],
                           peak_metrics['spacing_dev'][~peak_metrics['pass_spacing']],
                           c='red',marker='x',s=50,label="Fail Spacing(P)")
        if valley_metrics['x'].size>0:
            axs[2].scatter(valley_metrics['x'][valley_metrics['pass_spacing']],
                           valley_metrics['spacing_dev'][valley_metrics['pass_spacing']],
                           c='green',marker='v',s=30,alpha=0.7,label="Pass Spacing(V)");
            axs[2].scatter(valley_metrics['x'][~valley_metrics['pass_spacing']],
                           valley_metrics['spacing_dev'][~valley_metrics['pass_spacing']],
                           c='magenta',marker='x',s=50,label="Fail Spacing(V)")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label=f"Max Spacing Dev")
        axs[2].set_title(f"Norm Spacing Dev")
        axs[2].set_ylabel("Norm Spacing Dev")
        axs[2].legend(fontsize='small');axs[2].grid(True,ls=':')
        # axs[2].set_yscale('log')
        
        # Envelope Ratio Panel (Observed V_env/P_env)
        if peak_metrics['x'].size>0:
            valid_r_p = np.isfinite(peak_metrics['env_ratio_obs'])
            axs[3].scatter(peak_metrics['x'][peak_metrics['pass_ratio']&valid_r_p],
                           peak_metrics['env_ratio_obs'][peak_metrics['pass_ratio']&valid_r_p],
                           c='blue',marker='^',s=30,alpha=0.7,label="Pass Ratio(P)")
            axs[3].scatter(peak_metrics['x'][~peak_metrics['pass_ratio']&valid_r_p],
                           peak_metrics['env_ratio_obs'][~peak_metrics['pass_ratio']&valid_r_p],
                           c='red',marker='x',s=50,label="Fail Ratio(P)")
        if valley_metrics['x'].size>0:
            valid_r_v = np.isfinite(valley_metrics['env_ratio_obs'])
            axs[3].scatter(valley_metrics['x'][valley_metrics['pass_ratio']&valid_r_v],
                           valley_metrics['env_ratio_obs'][valley_metrics['pass_ratio']&valid_r_v],
                           c='green',marker='v',s=30,alpha=0.7,label="Pass Ratio(V)")
            axs[3].scatter(valley_metrics['x'][~valley_metrics['pass_ratio']&valid_r_v],
                           valley_metrics['env_ratio_obs'][~valley_metrics['pass_ratio']&valid_r_v],
                           c='magenta',marker='x',s=50,label="Fail Ratio(V)")
        axs[3].axhline(max_allowed_Venv_Penv_ratio,c='k',ls='--',label=f"Max V/P Ratio")
        axs[3].axhline(0,c='grey',ls=':',alpha=0.5)
        axs[3].axhline(1,c='grey',ls=':',alpha=0.5,label="V=P line")
        axs[3].set_title("Observed V_env/P_env Ratio")
        axs[3].set_ylabel("V_env/P_env")
        axs[3].legend(fontsize='small')
        axs[3].grid(True,ls=':')
        axs[3].set_xlabel("X-coord")
        axs[3].set_ylim(-0.1,1.2)

        plt.tight_layout(rect=[0,0,1,0.97]);plt.show()

    # --- Final Alternation & Return ---
    final_peaks_x_arr=np.array(kept_peaks_x);final_peaks_y_arr=np.array(kept_peaks_y)
    final_valleys_x_arr=np.array(kept_valleys_x);final_valleys_y_arr=np.array(kept_valleys_y)
    if final_peaks_x_arr.size>0 or final_valleys_x_arr.size>0:
        fpxa,fpya,fvxa,fvya=_ensure_alternation((final_peaks_x_arr,final_peaks_y_arr),(final_valleys_x_arr,final_valleys_y_arr))
    else: fpxa,fpya,fvxa,fvya=np.array([]),np.array([]),np.array([]),np.array([])
    
    plot_cand_x_comb=np.concatenate((plot_cand_peaks_x_alt,plot_cand_valleys_x_alt)) if plot_cand_peaks_x_alt.size>0 or plot_cand_valleys_x_alt.size>0 else np.array([])
    plot_cand_y_comb=np.concatenate((plot_cand_peaks_y_alt,plot_cand_valleys_y_alt)) if plot_cand_peaks_y_alt.size>0 or plot_cand_valleys_y_alt.size>0 else np.array([])
    plot_cand_types_comb=np.concatenate((np.ones_like(plot_cand_peaks_x_alt),-np.ones_like(plot_cand_valleys_x_alt))) if plot_cand_x_comb.size>0 else np.array([])
    if plot_cand_x_comb.size>0:
        sidx_plot=np.argsort(plot_cand_x_comb)
        pdata_orch=(None,None,None,plot_cand_x_comb[sidx_plot],plot_cand_y_comb[sidx_plot],plot_cand_types_comb[sidx_plot])
    else: pdata_orch=(None,None,None,np.array([]),np.array([]),np.array([]))
    return fpxa,fpya,fvxa,fvya, pdata_orch


def filter_lfc_extrema_v6_hybrid(
    peaks_x_cand, peaks_y_cand_refined, # Y-values are already refined from original data
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis, # Full original spectrum for context & noise
    # Parameters for Rules
    spacing_poly_degree=2, spacing_uncertainty_const=2.0, spacing_max_dev_factor=3.0,
    envelope_savgol_window=51, envelope_savgol_polyorder=3, # For Savgol envelope
    max_allowed_Venv_Penv_ratio = 0.35, # Simplified ratio threshold
    peak_snr_min_thresh_poisson=3.0, # S/N threshold specifically for peaks
    # Parameters for S/N calculation context
    noise_estimation_window_pixels=15, global_fallback_noise_std=None,
    # Parameters for DBSCAN (applied to survivors of rule-filter)
    dbscan_eps=0.5, dbscan_min_samples=5,
    min_candidates_for_dbscan=10,
    feature_scaling_options: dict = None, # For _calculate_features_for_extrema_v3 scaling
    # Control
    plot_filter_diagnostics=False,
    verbose=False
):
    # --- 0. Initial Setup & Alternation ---
    if global_fallback_noise_std is None:
        global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))

    current_peaks_x, current_peaks_y, \
    current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined),
        (valleys_x_cand, valleys_y_cand_refined)
    )
    if verbose: print(f"  HybridFilter Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")

    plot_cand_peaks_x_alt = np.copy(current_peaks_x); plot_cand_peaks_y_alt = np.copy(current_peaks_y)
    plot_cand_valleys_x_alt = np.copy(current_valleys_x); plot_cand_valleys_y_alt = np.copy(current_valleys_y)

    if not current_peaks_x.size and not current_valleys_x.size:
        print("  No peaks or valleys after initial alternation for hybrid filtering.")
        plot_data_placeholder = (None,None,None, np.array([]),np.array([]),np.array([]))
        return current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y, plot_data_placeholder

    # --- 1. Fit Global Spacing Models ---
    spacing_model_p, spacing_poly_trans_p = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v, spacing_poly_trans_v = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)

    # --- 2. Fit Global Envelope Models (using Savitzky-Golay) ---
    # Envelopes are calculated on original_x_axis grid
    peak_envelope_on_orig_x = None
    if current_peaks_x.size >= 2:
        peak_envelope_on_orig_x = fit_envelope_savgol(current_peaks_x, current_peaks_y, original_x_axis,
                                                      savgol_window_len_pixels=envelope_savgol_window,
                                                      savgol_polyorder=envelope_savgol_polyorder, verbose=verbose)
    if peak_envelope_on_orig_x is None and verbose: print("  Warning: Failed to create peak envelope.")

    valley_envelope_on_orig_x = None
    if current_valleys_x.size >= 2:
        valley_envelope_on_orig_x = fit_envelope_savgol(current_valleys_x, current_valleys_y, original_x_axis,
                                                        savgol_window_len_pixels=envelope_savgol_window,
                                                        savgol_polyorder=envelope_savgol_polyorder, verbose=verbose)
    if valley_envelope_on_orig_x is None and verbose: print("  Warning: Failed to create valley envelope.")


    # --- Rule-Based Pre-Filtering ---
    peak_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    valley_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}

    # Filter Peaks
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        peak_metrics['x'].append(xp); peak_metrics['y'].append(yp)
        pass_pos, pass_snr_p, pass_spacing_p, pass_ratio_p = True, True, True, True

        if yp <= 1e-9: # Peaks should be meaningfully positive (allowing for small positive baseline)
             pass_pos = False

        if pass_pos: # Only proceed if peak is positive
            # S/N for Peaks: (yp - V_env(xp)) / noise_approx
            V_env_at_xp = np.interp(xp, original_x_axis, valley_envelope_on_orig_x) if valley_envelope_on_orig_x is not None else np.median(original_y_axis)
            if not np.isfinite(V_env_at_xp): V_env_at_xp = np.median(original_y_axis) # Fallback for V_env
            
            signal_P = yp - V_env_at_xp
            if signal_P <= 0: current_snr_P = -1 # Peak not above its estimated valley floor
            else:
                noise_approx_P = np.sqrt(max(1, yp) + max(1, V_env_at_xp))
                current_snr_P = signal_P / noise_approx_P if noise_approx_P > 1e-9 else signal_P / 1e-9
            if current_snr_P < peak_snr_min_thresh_poisson: pass_snr_p = False
        else: current_snr_P = np.nan; pass_snr_p = False
        peak_metrics['snr'].append(current_snr_P)
        
        # Spacing
        current_spacing_dev_p = np.nan
        if spacing_model_p: # Spacing check
            # ... (Spacing logic as before) ...
            devs_norm_list = []
            if i > 0:
                s_bwd_act = xp-current_peaks_x[i-1]; x_mid_bwd=(xp+current_peaks_x[i-1])/2.0
                s_bwd_exp_arr = predict_spacing(x_mid_bwd, spacing_model_p, spacing_poly_trans_p)
                if s_bwd_exp_arr is not None and np.isfinite(s_bwd_exp_arr[0]): devs_norm_list.append(abs(s_bwd_act-s_bwd_exp_arr[0])/spacing_uncertainty_const)
            if i < len(current_peaks_x)-1:
                s_fwd_act = current_peaks_x[i+1]-xp; x_mid_fwd=(xp+current_peaks_x[i+1])/2.0
                s_fwd_exp_arr = predict_spacing(x_mid_fwd, spacing_model_p, spacing_poly_trans_p)
                if s_fwd_exp_arr is not None and np.isfinite(s_fwd_exp_arr[0]): devs_norm_list.append(abs(s_fwd_act-s_fwd_exp_arr[0])/spacing_uncertainty_const)
            if devs_norm_list: current_spacing_dev_p=np.mean(devs_norm_list);
            else: current_spacing_dev_p=0 
            if current_spacing_dev_p > spacing_max_dev_factor: pass_spacing_p=False
        peak_metrics['spacing_dev'].append(current_spacing_dev_p)
        
        # Envelope Ratio (V_env/P_env)
        obs_r_peak = np.nan
        if peak_envelope_on_orig_x is not None and valley_envelope_on_orig_x is not None:
            P_env_val = np.interp(xp, original_x_axis, peak_envelope_on_orig_x)
            V_env_val = np.interp(xp, original_x_axis, valley_envelope_on_orig_x)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val):
                if yp <= V_env_val + 1e-6 : pass_ratio_p = False # Peak must be clearly above valley envelope
                if pass_ratio_p and P_env_val > 1e-6 and P_env_val > V_env_val:
                    obs_r_peak = V_env_val / P_env_val
                    if obs_r_peak > max_allowed_Venv_Penv_ratio: pass_ratio_p = False
                elif pass_ratio_p and P_env_val <= V_env_val : pass_ratio_p = False # No modulation
                # Heuristic: if peak is too far below its own envelope
                if pass_ratio_p and P_env_val > (V_env_val + 3*global_fallback_noise_std) and yp < (V_env_val + 0.3*(P_env_val - V_env_val)):
                    pass_ratio_p = False
            else: pass_ratio_p = False # Envelope interp failed
        elif peak_envelope_on_orig_x is None or valley_envelope_on_orig_x is None: pass_ratio_p = True # Cannot test
        peak_metrics['env_ratio_obs'].append(obs_r_peak)
        # peak_metrics['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio)

        peak_metrics['pass_snr'].append(pass_pos and pass_snr_p)
        peak_metrics['pass_spacing'].append(pass_spacing_p)
        peak_metrics['pass_ratio'].append(pass_ratio_p)
        peak_metrics['pass_all_rules'].append(pass_pos and pass_snr_p and pass_spacing_p and pass_ratio_p)

    # Filter Valleys (No S/N, only spacing and envelope ratio)
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        valley_metrics['x'].append(xv); valley_metrics['y'].append(yv)
        pass_spacing_v, pass_ratio_v = True, True
        valley_metrics['snr'].append(np.nan); valley_metrics['pass_snr'].append(True) # Skip S/N

        # Spacing
        current_spacing_dev_v=np.nan
        if spacing_model_v: # Spacing for valleys
            # ... (Valley spacing logic as before, ensure s_e_arr is handled if None) ...
            devs_v=[]; s_bwd_exp_val_v,s_fwd_exp_val_v=np.nan,np.nan
            if i > 0:
                s_b_act=xv-current_valleys_x[i-1];xm_b=(xv+current_valleys_x[i-1])/2
                s_e_arr=predict_spacing(xm_b,spacing_model_v,spacing_poly_trans_v)
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):s_bwd_exp_val_v=s_e_arr[0]
                if np.isfinite(s_bwd_exp_val_v):devs_v.append(abs(s_b_act-s_bwd_exp_val_v)/spacing_uncertainty_const)
            if i < len(current_valleys_x)-1:
                s_f_act=current_valleys_x[i+1]-xv;xm_f=(xv+current_valleys_x[i+1])/2
                s_e_arr=predict_spacing(xm_f,spacing_model_v,spacing_poly_trans_v)
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):s_fwd_exp_val_v=s_e_arr[0]
                if np.isfinite(s_fwd_exp_val_v):devs_v.append(abs(s_f_act-s_fwd_exp_val_v)/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v=np.mean(devs_v)
            else:current_spacing_dev_v=0
            if current_spacing_dev_v > spacing_max_dev_factor: pass_spacing_v=False
        valley_metrics['spacing_dev'].append(current_spacing_dev_v)
        
        # Envelope Ratio
        obs_r_valley = np.nan
        if peak_envelope_on_orig_x is not None and valley_envelope_on_orig_x is not None:
            P_env_val = np.interp(xv, original_x_axis, peak_envelope_on_orig_x)
            V_env_val = np.interp(xv, original_x_axis, valley_envelope_on_orig_x)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val):
                if yv >= P_env_val: pass_ratio_v = False
                if pass_ratio_v and P_env_val > V_env_val and yv > (V_env_val + 0.7 * (P_env_val - V_env_val)): pass_ratio_v=False # Heuristic
                if pass_ratio_v and P_env_val > 1e-6 and P_env_val > V_env_val:
                    obs_r_valley = V_env_val / P_env_val
                    if obs_r_valley > max_allowed_Venv_Penv_ratio: pass_ratio_v = False
                elif pass_ratio_v and P_env_val <= V_env_val: pass_ratio_v = False
            else: pass_ratio_v = False
        elif peak_envelope_on_orig_x is None or valley_envelope_on_orig_x is None : pass_ratio_v = True
        valley_metrics['env_ratio_obs'].append(obs_r_valley)
        # valley_metrics['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio)

        valley_metrics['pass_spacing'].append(pass_spacing_v); valley_metrics['pass_ratio'].append(pass_ratio_v)
        valley_metrics['pass_all_rules'].append(pass_spacing_v and pass_ratio_v)

    # Apply the pass_all_rules flags
    peak_metrics_np = {k: np.array(v) for k, v in peak_metrics.items() if v} # Handle potentially empty metric lists
    valley_metrics_np = {k: np.array(v) for k, v in valley_metrics.items() if v}

    # Check if 'pass_all_rules' key exists and has elements before indexing
    rule_kept_peaks_x = peak_metrics_np['x'][peak_metrics_np['pass_all_rules']] if 'pass_all_rules' in peak_metrics_np and peak_metrics_np['pass_all_rules'].size > 0 else np.array([])
    rule_kept_peaks_y = peak_metrics_np['y'][peak_metrics_np['pass_all_rules']] if 'pass_all_rules' in peak_metrics_np and peak_metrics_np['pass_all_rules'].size > 0 else np.array([])
    rule_kept_valleys_x = valley_metrics_np['x'][valley_metrics_np['pass_all_rules']] if 'pass_all_rules' in valley_metrics_np and valley_metrics_np['pass_all_rules'].size > 0 else np.array([])
    rule_kept_valleys_y = valley_metrics_np['y'][valley_metrics_np['pass_all_rules']] if 'pass_all_rules' in valley_metrics_np and valley_metrics_np['pass_all_rules'].size > 0 else np.array([])
    if verbose: print(f"  After rule-based pre-filter: {len(rule_kept_peaks_x)}P, {len(rule_kept_valleys_x)}V")

    # --- Diagnostic Plots for Rule-Based Filter ---
    if plot_filter_diagnostics:
        # ... (Plotting logic using peak_metrics_np and valley_metrics_np - ENSURE MARKER is used) ...
        # ... For Panel 1, predict_envelope now does not take x_scaler
        fig, axs = plt.subplots(4,1,figsize=(15,16),sharex=True);fig.suptitle("Rule-Based Filter Diagnostics (Savgol Env)",fontsize=16)
        xpr = np.linspace(np.min(original_x_axis),np.max(original_x_axis),300) if original_x_axis.size>0 else np.array([])
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig");
        if peak_envelope_on_orig_x is not None: axs[0].plot(original_x_axis, peak_envelope_on_orig_x,'b-',alpha=0.6,lw=1.5,label="P Env (Savgol)")
        if valley_envelope_on_orig_x is not None: axs[0].plot(original_x_axis, valley_envelope_on_orig_x,'g-',alpha=0.6,lw=1.5,label="V Env (Savgol)")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',marker='^',s=20,alpha=0.5,label="P Cands");axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if rule_kept_peaks_x.size>0:axs[0].scatter(rule_kept_peaks_x,rule_kept_peaks_y,c='blue',marker='^',s=40,label="Rule-Kept P")
        if rule_kept_valleys_x.size>0:axs[0].scatter(rule_kept_valleys_x,rule_kept_valleys_y,c='darkgreen',marker='v',s=40,label="Rule-Kept V")
        axs[0].set_title("Data, Savgol Envelopes, Filtered Extrema");axs[0].set_ylabel("Flux");axs[0].legend(fontsize='small',loc='upper right');axs[0].grid(True,ls=':')
        # S/N Panel
        if peak_metrics_np.get('x',np.array([])).size>0:axs[1].scatter(peak_metrics_np['x'][peak_metrics_np['pass_snr']],peak_metrics_np['snr'][peak_metrics_np['pass_snr']],c='blue',marker='^',s=30,alpha=0.7,label="Pass S/N(P)");axs[1].scatter(peak_metrics_np['x'][~peak_metrics_np['pass_snr']],peak_metrics_np['snr'][~peak_metrics_np['pass_snr']],c='red',marker='x',s=50,label="Fail S/N(P)")
        axs[1].axhline(peak_snr_min_thresh_poisson,c='k',ls='--',label=f"Peak S/N Thresh");axs[1].set_title("Peak S/N ((yp - Venv) / noise_approx)");axs[1].set_ylabel("S/N Value");axs[1].legend(fontsize='small');axs[1].grid(True,ls=':');axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        # Spacing Panel
        if peak_metrics_np.get('x',np.array([])).size>0:axs[2].scatter(peak_metrics_np['x'][peak_metrics_np['pass_spacing']],peak_metrics_np['spacing_dev'][peak_metrics_np['pass_spacing']],c='blue',marker='^',s=30,alpha=0.7,label="Pass Spacing(P)");axs[2].scatter(peak_metrics_np['x'][~peak_metrics_np['pass_spacing']],peak_metrics_np['spacing_dev'][~peak_metrics_np['pass_spacing']],c='red',marker='x',s=50,label="Fail Spacing(P)")
        if valley_metrics_np.get('x',np.array([])).size>0:axs[2].scatter(valley_metrics_np['x'][valley_metrics_np['pass_spacing']],valley_metrics_np['spacing_dev'][valley_metrics_np['pass_spacing']],c='green',marker='v',s=30,alpha=0.7,label="Pass Spacing(V)");axs[2].scatter(valley_metrics_np['x'][~valley_metrics_np['pass_spacing']],valley_metrics_np['spacing_dev'][~valley_metrics_np['pass_spacing']],c='magenta',marker='x',s=50,label="Fail Spacing(V)")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label=f"Max Spacing Dev");axs[2].set_title(f"Norm Spacing Dev");axs[2].set_ylabel("Norm Spacing Dev");axs[2].legend(fontsize='small');axs[2].grid(True,ls=':');axs[2].set_yscale('log')
        # Envelope Ratio Panel
        if peak_metrics_np.get('x',np.array([])).size>0:valid_r_p=np.isfinite(peak_metrics_np['env_ratio_obs']);axs[3].scatter(peak_metrics_np['x'][peak_metrics_np['pass_ratio']&valid_r_p],peak_metrics_np['env_ratio_obs'][peak_metrics_np['pass_ratio']&valid_r_p],c='blue',marker='^',s=30,alpha=0.7,label="Pass Ratio(P)");axs[3].scatter(peak_metrics_np['x'][~peak_metrics_np['pass_ratio']&valid_r_p],peak_metrics_np['env_ratio_obs'][~peak_metrics_np['pass_ratio']&valid_r_p],c='red',marker='x',s=50,label="Fail Ratio(P)")
        if valley_metrics_np.get('x',np.array([])).size>0:valid_r_v=np.isfinite(valley_metrics_np['env_ratio_obs']);axs[3].scatter(valley_metrics_np['x'][valley_metrics_np['pass_ratio']&valid_r_v],valley_metrics_np['env_ratio_obs'][valley_metrics_np['pass_ratio']&valid_r_v],c='green',marker='v',s=30,alpha=0.7,label="Pass Ratio(V)");axs[3].scatter(valley_metrics_np['x'][~valley_metrics_np['pass_ratio']&valid_r_v],valley_metrics_np['env_ratio_obs'][~valley_metrics_np['pass_ratio']&valid_r_v],c='magenta',marker='x',s=50,label="Fail Ratio(V)")
        axs[3].axhline(max_allowed_Venv_Penv_ratio,c='k',ls='--',label=f"Max V/P Ratio");axs[3].axhline(0,c='grey',ls=':',alpha=0.5);axs[3].axhline(1,c='grey',ls=':',alpha=0.5,label="V=P line");axs[3].set_title("Observed V_env/P_env Ratio");axs[3].set_ylabel("V_env/P_env");axs[3].legend(fontsize='small');axs[3].grid(True,ls=':');axs[3].set_xlabel("X-coord");axs[3].set_ylim(-0.1,1.2)
        plt.tight_layout(rect=[0,0,1,0.97]);plt.show()


    # --- DBSCAN Stage on Rule-Filtered Candidates ---
    # ... (DBSCAN logic from filter_lfc_extrema_v6_hybrid in previous response, using rule_kept_peaks/valleys) ...
    # ... This will use _calculate_features_for_extrema_v3 with the rule-survivors.
    dbscan_input_peaks_x, dbscan_input_peaks_y = rule_kept_peaks_x, rule_kept_peaks_y
    dbscan_input_valleys_x, dbscan_input_valleys_y = rule_kept_valleys_x, rule_kept_valleys_y
    _,_,_,_, all_dbscan_cand_x, all_dbscan_cand_y, all_dbscan_cand_types = \
        _ensure_alternation_and_tag(dbscan_input_peaks_x, dbscan_input_peaks_y, dbscan_input_valleys_x, dbscan_input_valleys_y)
    final_peaks_x_arr, final_peaks_y_arr = rule_kept_peaks_x, rule_kept_peaks_y
    final_valleys_x_arr, final_valleys_y_arr = rule_kept_valleys_x, rule_kept_valleys_y
    plot_data_for_orchestrator = (None,None,None,np.copy(all_dbscan_cand_x),np.copy(all_dbscan_cand_y),np.copy(all_dbscan_cand_types))

    if all_dbscan_cand_x.size >= min_candidates_for_dbscan:
        if verbose: print(f"  Applying DBSCAN to {all_dbscan_cand_x.size} rule-filtered candidates...")
        features_for_dbscan_unscaled = _calculate_features_for_extrema_v3(all_dbscan_cand_x,all_dbscan_cand_y,all_dbscan_cand_types,original_y_axis,original_x_axis,noise_estimation_window_pixels,global_fallback_noise_std,spacing_poly_degree,spacing_uncertainty_const)
        scaled_dbscan_features,dbscan_labels=None,None;dbscan_clustered_x,dbscan_clustered_y,dbscan_clustered_types=np.array([]),np.array([]),np.array([])
        if features_for_dbscan_unscaled.shape[0]>0 and features_for_dbscan_unscaled.shape[1]==3:
            current_scaling_options=feature_scaling_options if feature_scaling_options is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
            def get_s(sname):
                if sname=='minmax':return MinMaxScaler();
                if sname=='standard':return StandardScaler();
                if sname=='robust':return RobustScaler();
                raise ValueError(f"Unk scaler: {sname}")
            scols=[]
            try:
                if 'x' in current_scaling_options and features_for_dbscan_unscaled[:,0:1].size>0:scols.append(get_s(current_scaling_options['x']).fit_transform(features_for_dbscan_unscaled[:,0:1]))
                if 'prom_snr' in current_scaling_options and features_for_dbscan_unscaled[:,1:2].size>0:scols.append(get_s(current_scaling_options['prom_snr']).fit_transform(features_for_dbscan_unscaled[:,1:2]))
                if 'spacing_dev' in current_scaling_options and features_for_dbscan_unscaled[:,2:3].size>0:scols.append(get_s(current_scaling_options['spacing_dev']).fit_transform(features_for_dbscan_unscaled[:,2:3]))
                if not scols or len(scols)!=features_for_dbscan_unscaled.shape[1]:scaled_dbscan_features=RobustScaler().fit_transform(features_for_dbscan_unscaled) if features_for_dbscan_unscaled.size>0 else features_for_dbscan_unscaled
                else:scaled_dbscan_features=np.hstack(scols)
            except Exception as e:print(f"Err scaling for DBSCAN: {e}. Using unscaled.");scaled_dbscan_features=features_for_dbscan_unscaled
            if scaled_dbscan_features is not None and scaled_dbscan_features.shape[0]>=dbscan_min_samples:
                dbscan=DBSCAN(eps=dbscan_eps,min_samples=dbscan_min_samples);dbscan_labels=dbscan.fit_predict(scaled_dbscan_features)
                core_lbls=dbscan_labels[dbscan_labels!=-1]
                if core_lbls.size>0:
                    mc_lbl=Counter(core_lbls).most_common(1)[0][0];true_idx=np.where(dbscan_labels==mc_lbl)[0]
                    dbscan_clustered_x=all_dbscan_cand_x[true_idx];dbscan_clustered_y=all_dbscan_cand_y[true_idx];dbscan_clustered_types=all_dbscan_cand_types[true_idx]
            elif scaled_dbscan_features is not None:
                 if verbose:print(f"    Too few samples for DBSCAN. Keeping rule-survivors.")
                 dbscan_clustered_x,dbscan_clustered_y,dbscan_clustered_types=all_dbscan_cand_x,all_dbscan_cand_y,all_dbscan_cand_types
                 if scaled_dbscan_features.shape[0]>0:dbscan_labels=-np.ones(scaled_dbscan_features.shape[0],dtype=int)
        final_peaks_x_arr=dbscan_clustered_x[dbscan_clustered_types==1];final_peaks_y_arr=dbscan_clustered_y[dbscan_clustered_types==1]
        final_valleys_x_arr=dbscan_clustered_x[dbscan_clustered_types==-1];final_valleys_y_arr=dbscan_clustered_y[dbscan_clustered_types==-1]
        plot_data_for_orchestrator=(features_for_dbscan_unscaled if 'features_for_dbscan_unscaled' in locals() else None,scaled_dbscan_features if 'scaled_dbscan_features' in locals() else None,dbscan_labels if 'dbscan_labels' in locals() else None,np.copy(all_dbscan_cand_x),np.copy(all_dbscan_cand_y),np.copy(all_dbscan_cand_types))
    else:
        if verbose: print(f"  Skipping DBSCAN (only {all_dbscan_cand_x.size} candidates after rules).")
        # plot_data_for_orchestrator remains based on rule-filter inputs (all_dbscan_cand_x/y/types are rule_kept here)

    # --- Final Alternation & Return ---
    if final_peaks_x_arr.size > 0 or final_valleys_x_arr.size > 0:
        fpxa,fpya,fvxa,fvya=_ensure_alternation((final_peaks_x_arr,final_peaks_y_arr),(final_valleys_x_arr,final_valleys_y_arr))
    else: fpxa,fpya,fvxa,fvya=np.array([]),np.array([]),np.array([]),np.array([])
    return fpxa,fpya,fvxa,fvya, plot_data_for_orchestrator

def filter_lfc_extrema_v7_hybrid_sep_dbscan(
    peaks_x_cand, peaks_y_cand_refined,
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis,
    # Rule parameters
    spacing_poly_degree=2, spacing_uncertainty_const=2.0, spacing_max_dev_factor=3.0,
    envelope_savgol_window=51, envelope_savgol_polyorder=3,
    max_allowed_Venv_Penv_ratio = 0.35, # Increased slightly from 0.3
    peak_snr_min_thresh_poisson=3.0, # Adjusted from 5.0
    # DBSCAN & its feature parameters (now with _peaks and _valleys suffixes)
    dbscan_eps_peaks=0.5, dbscan_min_samples_peaks=5,
    dbscan_eps_valleys=0.5, dbscan_min_samples_valleys=5,
    min_candidates_for_dbscan=10, # Min rule-survivors to attempt DBSCAN
    feature_scaling_options_peaks: dict = None,
    feature_scaling_options_valleys: dict = None,
    spacing_poly_degree_feat=2, spacing_uncertainty_const_feat=2.0, # For DBSCAN features
    # Common context params
    noise_estimation_window_pixels=15, global_fallback_noise_std=None,
    plot_filter_diagnostics=False, verbose=False
):
    # --- 0. Initial Setup & Alternation (as in v6) ---
    # ... (Identical to filter_lfc_extrema_v6_hybrid) ...
    if global_fallback_noise_std is None: global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))
    current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined), (valleys_x_cand, valleys_y_cand_refined))
    if verbose: print(f"  HybridFilter_SepDBSCAN Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")
    plot_cand_peaks_x_alt = np.copy(current_peaks_x); plot_cand_peaks_y_alt = np.copy(current_peaks_y)
    plot_cand_valleys_x_alt = np.copy(current_valleys_x); plot_cand_valleys_y_alt = np.copy(current_valleys_y)
    if not current_peaks_x.size and not current_valleys_x.size: # Adjusted check: allow one type to be empty for rules if other exists
        print("  Not enough initial peaks AND valleys for full rule-based filtering. Applying simplified or skipping.")
        # If one is empty, rule-based filter might not be as effective, DBSCAN might be main filter.
        # For now, let's proceed if at least one type has candidates.
        # The DBSCAN part below will handle empty inputs for P or V separately.
        if not current_peaks_x.size and not current_valleys_x.size:
            plot_data_ph = (None,None,None,np.array([]),np.array([]),np.array([]))
            return current_peaks_x,current_peaks_y,current_valleys_x,current_valleys_y,plot_data_ph


    # --- 1. Rule-Based Pre-Filtering (as in v6, but using distinct spacing/envelope params) ---
    # Fit Spacing Models (using _rule params)
    spacing_model_p_rule, spacing_poly_trans_p_rule = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v_rule, spacing_poly_trans_v_rule = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)
    # Fit Envelope Models (using Savgol)
    peak_envelope_y = fit_envelope_savgol(current_peaks_x,current_peaks_y,original_x_axis,envelope_savgol_window,envelope_savgol_polyorder,verbose)
    valley_envelope_y = fit_envelope_savgol(current_valleys_x,current_valleys_y,original_x_axis,envelope_savgol_window,envelope_savgol_polyorder,verbose)

    # Initialize metrics for rule-based filter plotting
    peak_metrics_rule = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    valley_metrics_rule = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    
    rule_kept_peaks_x, rule_kept_peaks_y = [], []
    # Filter Peaks with Rules
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        # ... (Rule logic for peaks as in filter_lfc_extrema_v6_hybrid, using _rule params) ...
        # ... (Populate peak_metrics_rule) ...
        peak_metrics_rule['x'].append(xp); peak_metrics_rule['y'].append(yp)
        pass_pos, pass_snr_p, pass_spacing_p, pass_ratio_p = True,True,True,True
        if yp<=1e-9:pass_pos=False
        if pass_pos:
            V_env_xp=np.interp(xp,original_x_axis,valley_envelope_y) if valley_envelope_y is not None else np.median(original_y_axis)
            if not np.isfinite(V_env_xp):V_env_xp=np.median(original_y_axis)
            signal_P=yp-V_env_xp; signal_P=max(0,signal_P)
            noise_approx_P=np.sqrt(max(1,yp)+max(1,V_env_xp)); current_snr_P=signal_P/noise_approx_P if noise_approx_P>1e-9 else signal_P/1e-9
            if current_snr_P<peak_snr_min_thresh_poisson:pass_snr_p=False
        else:current_snr_P=np.nan;pass_snr_p=False
        peak_metrics_rule['snr'].append(current_snr_P)
        current_spacing_dev_p=np.nan
        if spacing_model_p_rule:
            devs=[];s_b_e,s_f_e=np.nan,np.nan
            if i>0:
                s_b_a=xp-current_peaks_x[i-1]
                xm_b=(xp+current_peaks_x[i-1])/2
                s_b_e_arr=predict_spacing(xm_b,spacing_model_p_rule,spacing_poly_trans_p_rule)
                if s_b_e_arr is not None and np.isfinite(s_b_e_arr[0]):
                    s_b_e=s_b_e_arr[0]
                    devs.append(abs(s_b_a-s_b_e)/spacing_uncertainty_const)
            if i<len(current_peaks_x)-1:
                s_f_a=current_peaks_x[i+1]-xp
                xm_f=(xp+current_peaks_x[i+1])/2;
                s_f_e_arr=predict_spacing(xm_f,spacing_model_p_rule,spacing_poly_trans_p_rule);
                if s_f_e_arr is not None and np.isfinite(s_f_e_arr[0]):
                    s_f_e=s_f_e_arr[0];
                    devs.append(abs(s_f_a-s_f_e)/spacing_uncertainty_const)
            if devs:current_spacing_dev_p=np.mean(devs)
            else:current_spacing_dev_p=0
            if current_spacing_dev_p>spacing_max_dev_factor:pass_spacing_p=False
        peak_metrics_rule['spacing_dev'].append(current_spacing_dev_p)
        obs_r_p=np.nan
        if peak_envelope_y is not None and valley_envelope_y is not None:
            P_env_v=np.interp(xp,original_x_axis,peak_envelope_y);V_env_v=np.interp(xp,original_x_axis,valley_envelope_y)
            if np.isfinite(P_env_v) and np.isfinite(V_env_v):
                if yp<=V_env_v:pass_ratio_p=False
                if pass_ratio_p and P_env_v>1e-6 and P_env_v>V_env_v:
                    obs_r_p=V_env_v/P_env_v;
                    if obs_r_p>max_allowed_Venv_Penv_ratio:
                        pass_ratio_p=False
                elif pass_ratio_p and P_env_v<=V_env_v:pass_ratio_p=False
            else:pass_ratio_p=False
        elif peak_envelope_y is None or valley_envelope_y is None:pass_ratio_p=True
        peak_metrics_rule['env_ratio_obs'].append(obs_r_p);
        # peak_metrics_rule['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio)
        peak_metrics_rule['pass_snr'].append(pass_pos and pass_snr_p);peak_metrics_rule['pass_spacing'].append(pass_spacing_p);peak_metrics_rule['pass_ratio'].append(pass_ratio_p)
        if pass_pos and pass_snr_p and pass_spacing_p and pass_ratio_p:rule_kept_peaks_x.append(xp);rule_kept_peaks_y.append(yp);peak_metrics_rule['pass_all_rules'].append(True)
        else:peak_metrics_rule['pass_all_rules'].append(False)
    rule_kept_peaks_x = np.array(rule_kept_peaks_x); rule_kept_peaks_y = np.array(rule_kept_peaks_y)

    # Filter Valleys with Rules (No S/N)
    rule_kept_valleys_x, rule_kept_valleys_y = [], []
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        # ... (Rule logic for valleys as in v6, populating valley_metrics_rule) ...
        valley_metrics_rule['x'].append(xv);valley_metrics_rule['y'].append(yv);pass_spacing_v,pass_ratio_v=True,True
        valley_metrics_rule['snr'].append(np.nan);valley_metrics_rule['pass_snr'].append(True)
        current_spacing_dev_v=np.nan
        if spacing_model_v_rule:
            devs_v=[];s_b_e_v,s_f_e_v=np.nan,np.nan
            if i>0:
                s_b_a=xv-current_valleys_x[i-1];
                xm_b=(xv+current_valleys_x[i-1])/2;
                s_e_arr=predict_spacing(xm_b,spacing_model_v_rule,spacing_poly_trans_v_rule);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_b_e_v=s_e_arr[0];
                    devs_v.append(abs(s_b_a-s_b_e_v)/spacing_uncertainty_const)
            if i<len(current_valleys_x)-1:
                s_f_a=current_valleys_x[i+1]-xv;
                xm_f=(xv+current_valleys_x[i+1])/2;
                s_e_arr=predict_spacing(xm_f,spacing_model_v_rule,spacing_poly_trans_v_rule);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_f_e_v=s_e_arr[0];
                    devs_v.append(abs(s_f_a-s_f_e_v)/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v=np.mean(devs_v)
            else:current_spacing_dev_v=0
            if current_spacing_dev_v>spacing_max_dev_factor:pass_spacing_v=False
        valley_metrics_rule['spacing_dev'].append(current_spacing_dev_v)
        obs_r_v=np.nan
        if peak_envelope_y is not None and valley_envelope_y is not None:
            P_env_v=np.interp(xv,original_x_axis,peak_envelope_y);V_env_v=np.interp(xv,original_x_axis,valley_envelope_y)
            if np.isfinite(P_env_v) and np.isfinite(V_env_v):
                if yv>=P_env_v:pass_ratio_v=False
                if pass_ratio_v and P_env_v>1e-6 and P_env_v>V_env_v:
                    obs_r_v=V_env_v/P_env_v;
                    if obs_r_v>max_allowed_Venv_Penv_ratio:
                        pass_ratio_v=False
                elif pass_ratio_v and P_env_v<=V_env_v:pass_ratio_v=False
            else:pass_ratio_v=False
        elif peak_envelope_y is None or valley_envelope_y is None:pass_ratio_v=True
        valley_metrics_rule['env_ratio_obs'].append(obs_r_v);
        # valley_metrics_rule['env_ratio_exp'].append(max_allowed_Venv_Penv_ratio)
        valley_metrics_rule['pass_spacing'].append(pass_spacing_v);
        valley_metrics_rule['pass_ratio'].append(pass_ratio_v)
        if pass_spacing_v and pass_ratio_v:
            rule_kept_valleys_x.append(xv);
            rule_kept_valleys_y.append(yv);
            valley_metrics_rule['pass_all_rules'].append(True)
        else:valley_metrics_rule['pass_all_rules'].append(False)
    rule_kept_valleys_x = np.array(rule_kept_valleys_x); rule_kept_valleys_y = np.array(rule_kept_valleys_y)

    for key in peak_metrics_rule: peak_metrics_rule[key] = np.array(peak_metrics_rule[key])
    for key in valley_metrics_rule: valley_metrics_rule[key] = np.array(valley_metrics_rule[key])
    
    if plot_filter_diagnostics: # Plot for rule-based stage
        # ... (Identical plotting logic for 4 panels as in filter_lfc_extrema_v5, using peak_metrics_rule, valley_metrics_rule)
        fig,axs=plt.subplots(4,1,figsize=(15,16),sharex=True);fig.suptitle("Rule-Based Pre-Filter Diagnostics",fontsize=16)
        xpr=np.linspace(np.min(original_x_axis),np.max(original_x_axis),300) if original_x_axis.size>0 else np.array([])
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig");
        if peak_envelope_y is not None:axs[0].plot(original_x_axis,peak_envelope_y,'b-',alpha=0.6,lw=1.5,label="P Env")
        if valley_envelope_y is not None:axs[0].plot(original_x_axis,valley_envelope_y,'g-',alpha=0.6,lw=1.5,label="V Env")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',marker='^',s=20,alpha=0.5,label="P Cands");axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if peak_metrics_rule.get('x',np.array([])).size>0 and peak_metrics_rule.get('pass_all_rules',np.array([])).size>0:axs[0].scatter(peak_metrics_rule['x'][peak_metrics_rule['pass_all_rules']],peak_metrics_rule['y'][peak_metrics_rule['pass_all_rules']],c='blue',marker='^',s=40,label="Rule-Kept P")
        if valley_metrics_rule.get('x',np.array([])).size>0 and valley_metrics_rule.get('pass_all_rules',np.array([])).size>0:axs[0].scatter(valley_metrics_rule['x'][valley_metrics_rule['pass_all_rules']],valley_metrics_rule['y'][valley_metrics_rule['pass_all_rules']],c='darkgreen',marker='v',s=40,label="Rule-Kept V")
        axs[0].set_title("Data, Envelopes, Rule-Filtered Extrema");axs[0].set_ylabel("Flux");axs[0].legend(fontsize='small',loc='upper right');axs[0].grid(True,ls=':')
        if peak_metrics_rule.get('x',np.array([])).size>0:axs[1].scatter(peak_metrics_rule['x'][peak_metrics_rule['pass_snr']],peak_metrics_rule['snr'][peak_metrics_rule['pass_snr']],c='b',marker='^',s=30,alpha=0.7,label="P S/N Pass");axs[1].scatter(peak_metrics_rule['x'][~peak_metrics_rule['pass_snr']],peak_metrics_rule['snr'][~peak_metrics_rule['pass_snr']],c='r',marker='x',s=50,label="P S/N Fail")
        axs[1].axhline(peak_snr_min_thresh_poisson,c='k',ls='--',label="P S/N Thresh");axs[1].set_title("Peak S/N");axs[1].set_ylabel("S/N");axs[1].legend(fontsize='small');axs[1].grid(True,ls=':');axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        if peak_metrics_rule.get('x',np.array([])).size>0:axs[2].scatter(peak_metrics_rule['x'][peak_metrics_rule['pass_spacing']],peak_metrics_rule['spacing_dev'][peak_metrics_rule['pass_spacing']],c='b',marker='^',s=30,alpha=0.7,label="P Spacing Pass");axs[2].scatter(peak_metrics_rule['x'][~peak_metrics_rule['pass_spacing']],peak_metrics_rule['spacing_dev'][~peak_metrics_rule['pass_spacing']],c='r',marker='x',s=50,label="P Spacing Fail")
        if valley_metrics_rule.get('x',np.array([])).size>0:axs[2].scatter(valley_metrics_rule['x'][valley_metrics_rule['pass_spacing']],valley_metrics_rule['spacing_dev'][valley_metrics_rule['pass_spacing']],c='g',marker='v',s=30,alpha=0.7,label="V Spacing Pass");axs[2].scatter(valley_metrics_rule['x'][~valley_metrics_rule['pass_spacing']],valley_metrics_rule['spacing_dev'][~valley_metrics_rule['pass_spacing']],c='m',marker='x',s=50,label="V Spacing Fail")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label="Max Spacing Dev");axs[2].set_title("Norm Spacing Dev");axs[2].set_ylabel("Norm Spacing Dev");axs[2].legend(fontsize='small');axs[2].grid(True,ls=':');axs[2].set_yscale('log')
        if peak_metrics_rule.get('x',np.array([])).size>0:valid_rp=np.isfinite(peak_metrics_rule['env_ratio_obs']);axs[3].scatter(peak_metrics_rule['x'][peak_metrics_rule['pass_ratio']&valid_rp],peak_metrics_rule['env_ratio_obs'][peak_metrics_rule['pass_ratio']&valid_rp],c='b',marker='^',s=30,alpha=0.7,label="P Ratio Pass");axs[3].scatter(peak_metrics_rule['x'][~peak_metrics_rule['pass_ratio']&valid_rp],peak_metrics_rule['env_ratio_obs'][~peak_metrics_rule['pass_ratio']&valid_rp],c='r',marker='x',s=50,label="P Ratio Fail")
        if valley_metrics_rule.get('x',np.array([])).size>0:valid_rv=np.isfinite(valley_metrics_rule['env_ratio_obs']);axs[3].scatter(valley_metrics_rule['x'][valley_metrics_rule['pass_ratio']&valid_rv],valley_metrics_rule['env_ratio_obs'][valley_metrics_rule['pass_ratio']&valid_rv],c='g',marker='v',s=30,alpha=0.7,label="V Ratio Pass");axs[3].scatter(valley_metrics_rule['x'][~valley_metrics_rule['pass_ratio']&valid_rv],valley_metrics_rule['env_ratio_obs'][~valley_metrics_rule['pass_ratio']&valid_rv],c='m',marker='x',s=50,label="V Ratio Fail")
        axs[3].axhline(max_allowed_Venv_Penv_ratio,c='k',ls='--',label="Max V/P Ratio");axs[3].axhline(0,c='grey',ls=':',alpha=0.5);axs[3].axhline(1,c='grey',ls=':',alpha=0.5,label="V=P line");axs[3].set_title("Observed V_env/P_env Ratio");axs[3].set_ylabel("V_env/P_env");axs[3].legend(fontsize='small');axs[3].grid(True,ls=':');axs[3].set_xlabel("X-coord");axs[3].set_ylim(-0.1,1.2)
        plt.tight_layout(rect=[0,0,1,0.97]);plt.show()


    # --- DBSCAN Stage on Rule-Filtered Candidates ---
    # Default to rule-filtered if DBSCAN is skipped or fails
    final_dbscan_peaks_x, final_dbscan_peaks_y = rule_kept_peaks_x, rule_kept_peaks_y
    final_dbscan_valleys_x, final_dbscan_valleys_y = rule_kept_valleys_x, rule_kept_valleys_y
    
    # Prepare combined list of rule survivors for DBSCAN's feature calculation plotting
    # These are the candidates that *would be* fed to DBSCAN.
    # Their y-values are already refined.
    dbscan_cand_plot_x = np.concatenate((rule_kept_peaks_x, rule_kept_valleys_x)) if rule_kept_peaks_x.size > 0 or rule_kept_valleys_x.size > 0 else np.array([])
    dbscan_cand_plot_y = np.concatenate((rule_kept_peaks_y, rule_kept_valleys_y)) if rule_kept_peaks_y.size > 0 or rule_kept_valleys_y.size > 0 else np.array([])
    dbscan_cand_plot_types = np.concatenate((np.ones_like(rule_kept_peaks_x), -np.ones_like(rule_kept_valleys_x))) if dbscan_cand_plot_x.size > 0 else np.array([])
    if dbscan_cand_plot_x.size > 0:
        sidx_dbscan_plot = np.argsort(dbscan_cand_plot_x)
        plot_data_for_dbscan_summary = (None, None, None, # No features/labels YET for this particular set
                                      dbscan_cand_plot_x[sidx_dbscan_plot],
                                      dbscan_cand_plot_y[sidx_dbscan_plot],
                                      dbscan_cand_plot_types[sidx_dbscan_plot])
    else:
        plot_data_for_dbscan_summary = (None, None, None, np.array([]), np.array([]), np.array([]))


    # --- Separate DBSCAN for Peaks (survivors of rules) ---
    dbscan_peaks_plot_data = (None, None, None, np.array(rule_kept_peaks_x), np.array(rule_kept_peaks_y), np.ones_like(rule_kept_peaks_x)) # For plotting
    if rule_kept_peaks_x.size >= min_candidates_for_dbscan and rule_kept_valleys_x.size > 0: # Need valleys as opposites for prom S/N
        if verbose: print(f"  Applying DBSCAN to {len(rule_kept_peaks_x)} rule-filtered peaks...")
        features_peaks_unscaled = _calculate_features_for_one_type( # NEW helper
            rule_kept_peaks_x, rule_kept_peaks_y,
            rule_kept_valleys_x, rule_kept_valleys_y, # Use rule-kept valleys as opposites
            is_peak_type=True,
            original_y_data_for_noise=original_y_axis, original_x_data_for_noise=original_x_axis,
            noise_estimation_window_pixels=noise_estimation_window_pixels,
            global_fallback_noise_std=global_fallback_noise_std,
            spacing_poly_degree=spacing_poly_degree, # Use same spacing params as rules or dedicated?
            spacing_uncertainty_const=spacing_uncertainty_const
        )
        scaled_features_p, labels_p = None, None
        if features_peaks_unscaled.shape[0] > 0:
            # ... (Apply feature_scaling_options_peaks) ...
            current_scaling_opts_p = feature_scaling_options_peaks if feature_scaling_options_peaks is not None else {'x':'minmax','prom_snr':'standard','spacing_dev':'standard'}
            def get_s(sname):
                if sname=='minmax':return MinMaxScaler();
                if sname=='standard':return StandardScaler();
                if sname=='robust':return RobustScaler();
                raise ValueError(f"Unk scaler: {sname}")
            scols_p=[]
            try:
                if 'x' in current_scaling_opts_p and features_peaks_unscaled[:,0:1].size>0:scols_p.append(get_s(current_scaling_opts_p['x']).fit_transform(features_peaks_unscaled[:,0:1]))
                if 'prom_snr' in current_scaling_opts_p and features_peaks_unscaled[:,1:2].size>0:scols_p.append(get_s(current_scaling_opts_p['prom_snr']).fit_transform(features_peaks_unscaled[:,1:2]))
                if 'spacing_dev' in current_scaling_opts_p and features_peaks_unscaled[:,2:3].size>0:scols_p.append(get_s(current_scaling_opts_p['spacing_dev']).fit_transform(features_peaks_unscaled[:,2:3]))
                if not scols_p or len(scols_p)!=features_peaks_unscaled.shape[1]: scaled_features_p=RobustScaler().fit_transform(features_peaks_unscaled) if features_peaks_unscaled.size>0 else features_peaks_unscaled
                else: scaled_features_p=np.hstack(scols_p)
            except Exception as e: print(f"Err scaling peaks for DBSCAN: {e}. Using unscaled."); scaled_features_p=features_peaks_unscaled
            
            if scaled_features_p is not None and scaled_features_p.shape[0] >= dbscan_min_samples_peaks:
                dbscan_p = DBSCAN(eps=dbscan_eps_peaks, min_samples=dbscan_min_samples_peaks)
                labels_p = dbscan_p.fit_predict(scaled_features_p)
                core_lbls_p = labels_p[labels_p != -1]
                if core_lbls_p.size > 0:
                    mc_lbl_p = Counter(core_lbls_p).most_common(1)[0][0]
                    true_idx_p = np.where(labels_p == mc_lbl_p)[0]
                    final_dbscan_peaks_x = rule_kept_peaks_x[true_idx_p]
                    final_dbscan_peaks_y = rule_kept_peaks_y[true_idx_p]
                else: final_dbscan_peaks_x, final_dbscan_peaks_y = np.array([]), np.array([])
            elif scaled_features_p is not None: # Not enough for DBSCAN, keep all rule-survivors
                 if verbose: print(f"    Too few peak samples ({scaled_features_p.shape[0]}) for DBSCAN. Keeping all rule-surviving peaks.")
                 # final_dbscan_peaks_x, final_dbscan_peaks_y are already rule_kept_peaks_x/y
                 if scaled_features_p.shape[0] > 0: labels_p = -np.ones(scaled_features_p.shape[0], dtype=int)


        dbscan_peaks_plot_data = (features_peaks_unscaled if 'features_peaks_unscaled' in locals() else None,
                                  scaled_features_p, labels_p,
                                  np.copy(rule_kept_peaks_x), np.copy(rule_kept_peaks_y),
                                  np.ones_like(rule_kept_peaks_x)) # Types for plotting
    else:
        if verbose: print(f"  Skipping DBSCAN for peaks (only {rule_kept_peaks_x.size} candidates after rules or no valleys).")

    # --- Separate DBSCAN for Valleys (survivors of rules) ---
    # Valleys use the *output* of peak DBSCAN as their opposites
    dbscan_valleys_plot_data = (None, None, None, np.array(rule_kept_valleys_x), np.array(rule_kept_valleys_y), -np.ones_like(rule_kept_valleys_x))
    # if rule_kept_valleys_x.size >= min_candidates_for_dbscan and final_dbscan_peaks_x.size > 0: # Need peaks as opposites
    #     if verbose: print(f"  Applying DBSCAN to {len(rule_kept_valleys_x)} rule-filtered valleys...")
    #     features_valleys_unscaled = _calculate_features_for_one_type(
    #         rule_kept_valleys_x, rule_kept_valleys_y,
    #         final_dbscan_peaks_x, final_dbscan_peaks_y, # Use DBSCAN-filtered peaks as opposites
    #         is_peak_type=False,
    #         original_y_data_for_noise=original_y_axis, original_x_data_for_noise=original_x_axis,
    #         noise_estimation_window_pixels=noise_estimation_window_pixels,
    #         global_fallback_noise_std=global_fallback_noise_std,
    #         spacing_poly_degree=spacing_poly_degree, # Use same spacing params or dedicated?
    #         spacing_uncertainty_const=spacing_uncertainty_const
    #     )
    #     scaled_features_v, labels_v = None, None
    #     if features_valleys_unscaled.shape[0] > 0:
    #         # ... (Apply feature_scaling_options_valleys) ...
    #         current_scaling_opts_v = feature_scaling_options_valleys if feature_scaling_options_valleys is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
    #         scols_v=[] # Similar scaling logic as for peaks
    #         try:
    #             if 'x' in current_scaling_opts_v and features_valleys_unscaled[:,0:1].size>0:scols_v.append(get_s(current_scaling_opts_v['x']).fit_transform(features_valleys_unscaled[:,0:1]))
    #             if 'prom_snr' in current_scaling_opts_v and features_valleys_unscaled[:,1:2].size>0:scols_v.append(get_s(current_scaling_opts_v['prom_snr']).fit_transform(features_valleys_unscaled[:,1:2]))
    #             if 'spacing_dev' in current_scaling_opts_v and features_valleys_unscaled[:,2:3].size>0:scols_v.append(get_s(current_scaling_opts_v['spacing_dev']).fit_transform(features_valleys_unscaled[:,2:3]))
    #             if not scols_v or len(scols_v)!=features_valleys_unscaled.shape[1]: scaled_features_v=RobustScaler().fit_transform(features_valleys_unscaled) if features_valleys_unscaled.size>0 else features_valleys_unscaled
    #             else: scaled_features_v=np.hstack(scols_v)
    #         except Exception as e: print(f"Err scaling valleys for DBSCAN: {e}. Using unscaled."); scaled_features_v=features_valleys_unscaled

    #         if scaled_features_v is not None and scaled_features_v.shape[0] >= dbscan_min_samples_valleys:
    #             dbscan_v = DBSCAN(eps=dbscan_eps_valleys, min_samples=dbscan_min_samples_valleys)
    #             labels_v = dbscan_v.fit_predict(scaled_features_v)
    #             core_lbls_v = labels_v[labels_v != -1]
    #             if core_lbls_v.size > 0:
    #                 mc_lbl_v = Counter(core_lbls_v).most_common(1)[0][0]
    #                 true_idx_v = np.where(labels_v == mc_lbl_v)[0]
    #                 final_dbscan_valleys_x = rule_kept_valleys_x[true_idx_v]
    #                 final_dbscan_valleys_y = rule_kept_valleys_y[true_idx_v]
    #             else: final_dbscan_valleys_x, final_dbscan_valleys_y = np.array([]), np.array([])
    #         elif scaled_features_v is not None :
    #              if verbose: print(f"    Too few valley samples ({scaled_features_v.shape[0]}) for DBSCAN. Keeping all rule-surviving valleys.")
    #              if scaled_features_v.shape[0] > 0: labels_v = -np.ones(scaled_features_v.shape[0], dtype=int)
        
    #     dbscan_valleys_plot_data = (features_valleys_unscaled if 'features_valleys_unscaled' in locals() else None,
    #                                scaled_features_v, labels_v,
    #                                np.copy(rule_kept_valleys_x), np.copy(rule_kept_valleys_y),
    #                                -np.ones_like(rule_kept_valleys_x))
    # else:
    #     if verbose: print(f"  Skipping DBSCAN for valleys (only {rule_kept_valleys_x.size} candidates or no peaks from DBSCAN).")


    # --- Final Alternation & Return ---
    if final_dbscan_peaks_x.size > 0 or final_dbscan_valleys_x.size > 0:
        fpxa,fpya,fvxa,fvya=_ensure_alternation(
            (final_dbscan_peaks_x,final_dbscan_peaks_y),
            (final_dbscan_valleys_x,final_dbscan_valleys_y)
        )
    else: fpxa,fpya,fvxa,fvya=np.array([]),np.array([]),np.array([]),np.array([])
    
    # For the final summary plot, we want to show candidates that went into DBSCAN
    # (i.e. rule_kept) vs the final output from DBSCAN.
    # plot_data_for_orchestrator will be a tuple of two tuples for plotting:
    # (peak_plot_info, valley_plot_info)
    # where peak_plot_info = (unscaled_feat_p, scaled_feat_p, labels_p, rule_kept_p_x, rule_kept_p_y)
    
    plot_data_for_orchestrator = (dbscan_peaks_plot_data, dbscan_valleys_plot_data)

    return fpxa,fpya,fvxa,fvya, plot_data_for_orchestrator

def filter_lfc_extrema_v8(
    peaks_x_cand, peaks_y_cand_refined,
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis,
    # Rule parameters
    spacing_poly_degree=2, spacing_uncertainty_const=2.0, spacing_max_dev_factor=3.0,
    envelope_savgol_window=51, envelope_savgol_polyorder=3,
    ratio_trend_savgol_window=101, ratio_trend_savgol_polyorder=2, # For smoothing Venv/Penv ratio
    envelope_ratio_max_abs_dev_from_trend = 0.1, # Max deviation from smoothed R_exp(x)
    peak_snr_min_thresh_poisson=3.0,
    # DBSCAN & its feature parameters
    dbscan_eps_peaks=0.5, dbscan_min_samples_peaks=5,
    dbscan_eps_valleys=0.5, dbscan_min_samples_valleys=5,
    min_candidates_for_dbscan=10,
    feature_scaling_options_peaks: dict = None, feature_scaling_options_valleys: dict = None,
    spacing_poly_degree_feat=2, spacing_uncertainty_const_feat=2.0,
    # Common context params
    noise_estimation_window_pixels=15, global_fallback_noise_std=None,
    plot_filter_diagnostics=False, verbose=False
):
    # --- 0. Initial Setup & Alternation ---
    if global_fallback_noise_std is None:
        global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))
    current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined), (valleys_x_cand, valleys_y_cand_refined)
    )
    if verbose: print(f"  HybridFilter_SavgolRatio Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")
    plot_cand_peaks_x_alt, plot_cand_peaks_y_alt = np.copy(current_peaks_x), np.copy(current_peaks_y)
    plot_cand_valleys_x_alt, plot_cand_valleys_y_alt = np.copy(current_valleys_x), np.copy(current_valleys_y)
    if not current_peaks_x.size or not current_valleys_x.size:
        print("  Not enough P OR V after initial alternation for v8 filtering.")
        plot_data_ph = (None,None,None,np.array([]),np.array([]),np.array([])) # Peaks plot data
        plot_data_vh = (None,None,None,np.array([]),np.array([]),np.array([])) # Valleys plot data
        return current_peaks_x,current_peaks_y,current_valleys_x,current_valleys_y, (plot_data_ph, plot_data_vh)

    # --- 0.5 
    spacing_model_p, spacing_poly_trans_p = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v, spacing_poly_trans_v = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)

    # --- 1. Fit Savgol Envelopes (on original_x_axis grid) ---
    peak_envelope = fit_envelope_savgol(current_peaks_x, current_peaks_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)
    valley_envelope = fit_envelope_savgol(current_valleys_x, current_valleys_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)

    # --- 2. Calculate and Smooth the V_env(x) / P_env(x) Ratio Trend ---
    R_exp_trend = None
    if peak_envelope is not None and valley_envelope is not None and np.all(np.isfinite(peak_envelope)) and np.all(np.isfinite(valley_envelope)):
        # Avoid division by zero or very small P_env; add small epsilon or clip
        P_env_for_ratio = np.maximum(peak_envelope, 1e-6 * np.nanmax(peak_envelope) if np.nanmax(peak_envelope) > 0 else 1e-6)
        R_observed_globally = valley_envelope / P_env_for_ratio
        
        ratio_win = mathfunc.round_down_to_odd(int(ratio_trend_savgol_window))
        ratio_poly = int(ratio_trend_savgol_polyorder)
        if ratio_win <= ratio_poly: ratio_win = ratio_poly + 1 if ratio_poly % 2 == 0 else ratio_poly + 2; ratio_win = mathfunc.round_down_to_odd(ratio_win)
        
        if len(R_observed_globally) >= ratio_win:
            try:
                R_exp_trend = savgol_filter(R_observed_globally, window_length=ratio_win, polyorder=ratio_poly, mode="mirror")
            except ValueError as e:
                if verbose: print(f"  Warning: Savgol filter for ratio trend failed ({e}). Ratio check might be less effective.")
                R_exp_trend = R_observed_globally # Use unsmoothed ratio as fallback
        else:
            if verbose: print("  Warning: Not enough points to smooth Venv/Penv ratio trend. Using raw ratio.")
            R_exp_trend = R_observed_globally
    if R_exp_trend is None and verbose: print("  Warning: Could not compute R_exp_trend for envelope ratio criterion.")


    # --- Rule-Based Pre-Filtering Logic (Populate metrics dictionaries) ---
    # ... (Identical logic as in filter_lfc_extrema_v6_hybrid for S/N and Spacing) ...
    # ... The Envelope Ratio part will be modified to use R_exp_trend ...
    peak_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    valley_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    
    # Filter Peaks
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        peak_metrics['x'].append(xp); peak_metrics['y'].append(yp)
        pass_pos, pass_snr_p, pass_spacing_p, pass_ratio_p = True, True, True, True
        if yp <= 1e-9: pass_pos = False # Peaks must be positive
        current_snr_P_val=np.nan; current_spacing_dev_p_val=np.nan; obs_r_peak_val=np.nan; exp_r_peak_val=np.nan

        if pass_pos:
            V_env_at_xp = np.interp(xp, original_x_axis, valley_envelope) if valley_envelope is not None else np.median(original_y_axis)
            if not np.isfinite(V_env_at_xp): V_env_at_xp = np.median(original_y_axis)
            signal_P = yp - V_env_at_xp; signal_P = max(0, signal_P)
            noise_approx_P = np.sqrt(max(1,yp) + max(1,V_env_at_xp)); current_snr_P_val = signal_P/noise_approx_P if noise_approx_P>1e-9 else signal_P/1e-9
            if current_snr_P_val < peak_snr_min_thresh_poisson: pass_snr_p = False
        else: pass_snr_p = False
        peak_metrics['snr'].append(current_snr_P_val)

        if spacing_model_p: # Spacing logic
            devs=[];s_b_e,s_f_e=np.nan,np.nan
            if i>0:
                s_b_a=xp-current_peaks_x[i-1];
                xm_b=(xp+current_peaks_x[i-1])/2;
                s_b_e_arr=predict_spacing(xm_b,spacing_model_p,spacing_poly_trans_p)
                if s_b_e_arr is not None and np.isfinite(s_b_e_arr[0]):
                    s_b_e=s_b_e_arr[0];devs.append(abs(s_b_a-s_b_e)/spacing_uncertainty_const)
            if i<len(current_peaks_x)-1:
                s_f_a=current_peaks_x[i+1]-xp;
                xm_f=(xp+current_peaks_x[i+1])/2;
                s_f_e_arr=predict_spacing(xm_f,spacing_model_p,spacing_poly_trans_p);
                if s_f_e_arr is not None and np.isfinite(s_f_e_arr[0]):
                    s_f_e=s_f_e_arr[0];
                    devs.append(abs(s_f_a-s_f_e)/spacing_uncertainty_const)
            if devs:current_spacing_dev_p_val=np.mean(devs)
            else:current_spacing_dev_p_val=0 
            if current_spacing_dev_p_val > spacing_max_dev_factor: pass_spacing_p=False
        peak_metrics['spacing_dev'].append(current_spacing_dev_p_val)

        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None:
            P_env_val = np.interp(xp, original_x_axis, peak_envelope)
            V_env_val = np.interp(xp, original_x_axis, valley_envelope)
            exp_r_peak_val = np.interp(xp, original_x_axis, R_exp_trend)

            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_peak_val):
                if yp <= V_env_val : pass_ratio_p = False # Peak must be above local valley envelope
                # Heuristic: peak y should be somewhat close to its own envelope
                if pass_ratio_p and P_env_val > V_env_val and yp < (V_env_val + 0.3 * (P_env_val - V_env_val)): pass_ratio_p = False

                if pass_ratio_p and abs(P_env_val) > 1e-6:
                    obs_r_peak_val = V_env_val / P_env_val
                    if abs(obs_r_peak_val - exp_r_peak_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_p = False
                elif pass_ratio_p : pass_ratio_p = False # P_env near zero or V_env > P_env implicitly
            else: pass_ratio_p = False # Cannot evaluate envelopes or trend
        else: pass_ratio_p = True # Cannot test this criterion
        peak_metrics['env_ratio_obs'].append(obs_r_peak_val); peak_metrics['env_ratio_exp'].append(exp_r_peak_val)

        peak_metrics['pass_snr'].append(pass_pos and pass_snr_p); peak_metrics['pass_spacing'].append(pass_spacing_p); peak_metrics['pass_ratio'].append(pass_ratio_p)
        peak_metrics['pass_all_rules'].append(pass_pos and pass_snr_p and pass_spacing_p and pass_ratio_p)

    # Filter Valleys (No S/N, uses R_exp_trend for ratio check)
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        valley_metrics['x'].append(xv); valley_metrics['y'].append(yv)
        pass_spacing_v, pass_ratio_v = True, True
        valley_metrics['snr'].append(np.nan); valley_metrics['pass_snr'].append(True)
        current_spacing_dev_v_val=np.nan; obs_r_valley_val=np.nan; exp_r_valley_val=np.nan

        if spacing_model_v: # Spacing logic
            devs_v=[];s_b_e_v,s_f_e_v=np.nan,np.nan
            if i>0:
                s_b_a=xv-current_valleys_x[i-1];
                xm_b=(xv+current_valleys_x[i-1])/2;
                s_e_arr=predict_spacing(xm_b,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_b_e_v=s_e_arr[0];
                    devs_v.append(abs(s_b_a-s_b_e_v)/spacing_uncertainty_const)
            if i<len(current_valleys_x)-1:
                s_f_a=current_valleys_x[i+1]-xv;
                xm_f=(xv+current_valleys_x[i+1])/2;
                s_e_arr=predict_spacing(xm_f,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_f_e_v=s_e_arr[0];
                    devs_v.append(abs(s_f_a-s_f_e_v)/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v_val=np.mean(devs_v)
            else:current_spacing_dev_v_val=0
            if current_spacing_dev_v_val > spacing_max_dev_factor: pass_spacing_v=False
        valley_metrics['spacing_dev'].append(current_spacing_dev_v_val)

        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None:
            P_env_val = np.interp(xv, original_x_axis, peak_envelope)
            V_env_val = np.interp(xv, original_x_axis, valley_envelope)
            exp_r_valley_val = np.interp(xv, original_x_axis, R_exp_trend)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_valley_val):
                if yv >= P_env_val: pass_ratio_v = False # Valley must be below peak envelope
                if pass_ratio_v and P_env_val > V_env_val and yv > (V_env_val + 0.7*(P_env_val-V_env_val)): pass_ratio_v = False # Heuristic
                
                if pass_ratio_v and abs(P_env_val) > 1e-6:
                    obs_r_valley_val = V_env_val / P_env_val # Actual V_env/P_env at this valley's x
                    if abs(obs_r_valley_val - exp_r_valley_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_v = False
                elif pass_ratio_v: pass_ratio_v = False # P_env near zero or implies V_env > P_env
            else: pass_ratio_v = False
        else: pass_ratio_v = True
        valley_metrics['env_ratio_obs'].append(obs_r_valley_val); valley_metrics['env_ratio_exp'].append(exp_r_valley_val if 'exp_r_valley_val' in locals() and np.isfinite(exp_r_valley_val) else np.nan)
        
        valley_metrics['pass_spacing'].append(pass_spacing_v); valley_metrics['pass_ratio'].append(pass_ratio_v)
        valley_metrics['pass_all_rules'].append(pass_spacing_v and pass_ratio_v)

    # Apply pass_all_rules flags
    for metrics_dict in [peak_metrics, valley_metrics]:
        for key in metrics_dict: metrics_dict[key] = np.array(metrics_dict[key])
    
    rule_kept_peaks_x = peak_metrics['x'][peak_metrics['pass_all_rules']]
    rule_kept_peaks_y = peak_metrics['y'][peak_metrics['pass_all_rules']]
    rule_kept_valleys_x = valley_metrics['x'][valley_metrics['pass_all_rules']]
    rule_kept_valleys_y = valley_metrics['y'][valley_metrics['pass_all_rules']]
    if verbose: print(f"  After rule-based pre-filter: {len(rule_kept_peaks_x)}P, {len(rule_kept_valleys_x)}V")

    # --- Diagnostic Plots for Rule-Based Filter ---
    if plot_filter_diagnostics:
        # ... (Plotting logic as before, but Panel 4 shows |obs_ratio - R_exp_trend(x)|)
        fig, axs = plt.subplots(4,1,figsize=(15,16),sharex=True);fig.suptitle("Rule Filter Diagnostics (Savgol Ratio Trend)",fontsize=16)
        xpr = original_x_axis # Use original_x_axis for plotting envelopes and trend defined on it
        
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig");
        if peak_envelope is not None:axs[0].plot(original_x_axis,peak_envelope,'b-',alpha=0.6,lw=1.5,label="P Env")
        if valley_envelope is not None:axs[0].plot(original_x_axis,valley_envelope,'g-',alpha=0.6,lw=1.5,label="V Env")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',marker='^',s=20,alpha=0.5,label="P Cands");axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if rule_kept_peaks_x.size>0:axs[0].scatter(rule_kept_peaks_x,rule_kept_peaks_y,c='blue',marker='^',s=40,label="Rule-Kept P")
        if rule_kept_valleys_x.size>0:axs[0].scatter(rule_kept_valleys_x,rule_kept_valleys_y,c='darkgreen',marker='v',s=40,label="Rule-Kept V")
        axs[0].set_title("Data, Envelopes, Rule-Filtered");axs[0].set_ylabel("Flux");axs[0].legend(fontsize='small');axs[0].grid(True,ls=':')
        
        # S/N, Spacing panels as before using peak_metrics, valley_metrics
        if peak_metrics.get('x',np.array([])).size>0:axs[1].scatter(peak_metrics['x'][peak_metrics['pass_snr']],peak_metrics['snr'][peak_metrics['pass_snr']],c='b',marker='^',s=30,alpha=0.7,label="P S/N Pass");axs[1].scatter(peak_metrics['x'][~peak_metrics['pass_snr']],peak_metrics['snr'][~peak_metrics['pass_snr']],c='r',marker='x',s=50,label="P S/N Fail")
        axs[1].axhline(peak_snr_min_thresh_poisson,c='k',ls='--',label="P S/N Thresh");axs[1].set_title("Peak S/N");axs[1].set_ylabel("S/N");axs[1].legend(fontsize='small');axs[1].grid(True,ls=':');axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        
        if peak_metrics.get('x',np.array([])).size>0:axs[2].scatter(peak_metrics['x'][peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][peak_metrics['pass_spacing']],c='b',marker='^',s=30,alpha=0.7,label="P Spacing Pass");axs[2].scatter(peak_metrics['x'][~peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][~peak_metrics['pass_spacing']],c='r',marker='x',s=50,label="P Spacing Fail")
        if valley_metrics.get('x',np.array([])).size>0:axs[2].scatter(valley_metrics['x'][valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][valley_metrics['pass_spacing']],c='g',marker='v',s=30,alpha=0.7,label="V Spacing Pass");axs[2].scatter(valley_metrics['x'][~valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][~valley_metrics['pass_spacing']],c='m',marker='x',s=50,label="V Spacing Fail")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label="Max Spacing Dev");axs[2].set_title("Norm Spacing Dev");axs[2].set_ylabel("Norm Spacing Dev");axs[2].legend(fontsize='small');axs[2].grid(True,ls=':');axs[2].set_yscale('log')
        
        # Envelope Ratio Deviation Panel
        if R_exp_trend is not None:
            axs[3].plot(original_x_axis, R_exp_trend, 'k:', alpha=0.8, label="Smoothed Venv/Penv Trend (R_exp)")
        if peak_metrics.get('x',np.array([])).size > 0 :
            valid_p_ratios = np.isfinite(peak_metrics['env_ratio_obs']) & np.isfinite(peak_metrics['env_ratio_exp'])
            peak_ratio_devs = np.abs(peak_metrics['env_ratio_obs'][valid_p_ratios] - peak_metrics['env_ratio_exp'][valid_p_ratios])
            axs[3].scatter(peak_metrics['x'][valid_p_ratios][peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[peak_metrics['pass_ratio'][valid_p_ratios]], c='b', marker='^', s=30, alpha=0.7, label="P Ratio Pass")
            axs[3].scatter(peak_metrics['x'][valid_p_ratios][~peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[~peak_metrics['pass_ratio'][valid_p_ratios]],c='r', marker='x', s=50, label="P Ratio Fail")
        if valley_metrics.get('x',np.array([])).size > 0:
            valid_v_ratios = np.isfinite(valley_metrics['env_ratio_obs']) & np.isfinite(valley_metrics['env_ratio_exp'])
            valley_ratio_devs = np.abs(valley_metrics['env_ratio_obs'][valid_v_ratios] - valley_metrics['env_ratio_exp'][valid_v_ratios])
            axs[3].scatter(valley_metrics['x'][valid_v_ratios][valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[valley_metrics['pass_ratio'][valid_v_ratios]], c='g', marker='v', s=30, alpha=0.7, label="V Ratio Pass")
            axs[3].scatter(valley_metrics['x'][valid_v_ratios][~valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[~valley_metrics['pass_ratio'][valid_v_ratios]],c='m', marker='x', s=50, label="V Ratio Fail")
        axs[3].axhline(envelope_ratio_max_abs_dev_from_trend, c='k', ls='--', label="Max Ratio Dev")
        axs[3].set_title("Abs Dev from Smoothed Venv/Penv Ratio Trend");axs[3].set_ylabel("|Observed V/P - Smoothed V/P|");axs[3].legend(fontsize='small');axs[3].grid(True,ls=':');axs[3].set_xlabel("X-coord")
        
        plt.tight_layout(rect=[0,0,1,0.97]); plt.show()


    # --- DBSCAN Stage on Rule-Filtered Candidates ---
    # ... (Identical DBSCAN stage as in filter_lfc_extrema_v7_hybrid_sep_dbscan,
    #      using rule_kept_peaks_x/y and rule_kept_valleys_x/y as input) ...
    # ... It will call _calculate_features_for_one_type, scale, and run DBSCAN separately.
    dbscan_input_peaks_x, dbscan_input_peaks_y = rule_kept_peaks_x, rule_kept_peaks_y
    dbscan_input_valleys_x, dbscan_input_valleys_y = rule_kept_valleys_x, rule_kept_valleys_y
    final_peaks_x_arr, final_peaks_y_arr = rule_kept_peaks_x, rule_kept_peaks_y # Default if DBSCAN skipped
    final_valleys_x_arr, final_valleys_y_arr = rule_kept_valleys_x, rule_kept_valleys_y
    # Initialize plot data for orchestrator. This will be updated if DBSCAN runs.
    # Storing candidates that went INTO DBSCAN (i.e., rule_kept)
    dbscan_cand_plot_p_x = np.copy(rule_kept_peaks_x); dbscan_cand_plot_p_y = np.copy(rule_kept_peaks_y)
    dbscan_cand_plot_v_x = np.copy(rule_kept_valleys_x); dbscan_cand_plot_v_y = np.copy(rule_kept_valleys_y)
    dbscan_peaks_plot_info = (None,None,None, dbscan_cand_plot_p_x, dbscan_cand_plot_p_y, np.ones_like(dbscan_cand_plot_p_x))
    dbscan_valleys_plot_info = (None,None,None, dbscan_cand_plot_v_x, dbscan_cand_plot_v_y, -np.ones_like(dbscan_cand_plot_v_x))

    # Separate DBSCAN for Peaks
    if rule_kept_peaks_x.size >= min_candidates_for_dbscan and rule_kept_valleys_x.size > 0: # Need valleys as opposites
        if verbose:print(f"  Applying DBSCAN to {len(rule_kept_peaks_x)} rule-filtered peaks...")
        features_p_unscaled = _calculate_features_for_one_type(rule_kept_peaks_x,rule_kept_peaks_y,rule_kept_valleys_x,rule_kept_valleys_y,True,original_y_axis,original_x_axis,noise_estimation_window_pixels,global_fallback_noise_std,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
        scaled_fp,labels_p=None,None
        if features_p_unscaled.shape[0]>0:
            # ... (Feature specific scaling for peaks using feature_scaling_options_peaks) ...
            current_s_opts_p = feature_scaling_options_peaks if feature_scaling_options_peaks is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
            def get_s(sname):
                if sname=='minmax':return MinMaxScaler();
                if sname=='standard':return StandardScaler();
                if sname=='robust':return RobustScaler();
                raise ValueError(f"Unk scaler: {sname}")
            scols_p=[]
            try:
                if 'x' in current_s_opts_p and features_p_unscaled[:,0:1].size>0:scols_p.append(get_s(current_s_opts_p['x']).fit_transform(features_p_unscaled[:,0:1]))
                if 'prom_snr' in current_s_opts_p and features_p_unscaled[:,1:2].size>0:scols_p.append(get_s(current_s_opts_p['prom_snr']).fit_transform(features_p_unscaled[:,1:2]))
                if 'spacing_dev' in current_s_opts_p and features_p_unscaled[:,2:3].size>0:scols_p.append(get_s(current_s_opts_p['spacing_dev']).fit_transform(features_p_unscaled[:,2:3]))
                if not scols_p or len(scols_p)!=features_p_unscaled.shape[1]: scaled_fp=RobustScaler().fit_transform(features_p_unscaled) if features_p_unscaled.size>0 else features_p_unscaled
                else: scaled_fp=np.hstack(scols_p)
            except Exception as e: print(f"Err scaling peaks DBSCAN: {e}. Unscaled."); scaled_fp=features_p_unscaled

            if scaled_fp is not None and scaled_fp.shape[0]>=dbscan_min_samples_peaks:
                dbscan_p=DBSCAN(eps=dbscan_eps_peaks,min_samples=dbscan_min_samples_peaks);labels_p=dbscan_p.fit_predict(scaled_fp)
                core_lbl_p=labels_p[labels_p!=-1]
                if core_lbl_p.size>0:mc_lbl_p=Counter(core_lbl_p).most_common(1)[0][0];true_idx_p=np.where(labels_p==mc_lbl_p)[0];final_peaks_x_arr=rule_kept_peaks_x[true_idx_p];final_peaks_y_arr=rule_kept_peaks_y[true_idx_p]
                else:final_peaks_x_arr,final_peaks_y_arr=np.array([]),np.array([])
            elif scaled_fp is not None: # Not enough for DBSCAN
                 if verbose:print(f"    Too few peak samples for DBSCAN. Keeping rule-survivors.")
                 if scaled_fp.shape[0]>0: labels_p=-np.ones(scaled_fp.shape[0],dtype=int) # Mark all as outliers
        dbscan_peaks_plot_data = (features_p_unscaled if 'features_p_unscaled' in locals() else None, scaled_fp, labels_p, np.copy(rule_kept_peaks_x), np.copy(rule_kept_peaks_y), np.ones_like(rule_kept_peaks_x))
    
    # Separate DBSCAN for Valleys
    if rule_kept_valleys_x.size >= min_candidates_for_dbscan and final_peaks_x_arr.size > 0: # Use DBSCAN-filtered peaks as opposites
        if verbose:print(f"  Applying DBSCAN to {len(rule_kept_valleys_x)} rule-filtered valleys...")
        features_v_unscaled = _calculate_features_for_one_type(rule_kept_valleys_x,rule_kept_valleys_y,final_peaks_x_arr,final_peaks_y_arr,False,original_y_axis,original_x_axis,noise_estimation_window_pixels,global_fallback_noise_std,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
        scaled_fv,labels_v=None,None
        if features_v_unscaled.shape[0]>0:
            # ... (Feature specific scaling for valleys using feature_scaling_options_valleys) ...
            current_s_opts_v = feature_scaling_options_valleys if feature_scaling_options_valleys is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
            scols_v=[]
            try:
                if 'x' in current_s_opts_v and features_v_unscaled[:,0:1].size>0:scols_v.append(get_s(current_s_opts_v['x']).fit_transform(features_v_unscaled[:,0:1]))
                if 'prom_snr' in current_s_opts_v and features_v_unscaled[:,1:2].size>0:scols_v.append(get_s(current_s_opts_v['prom_snr']).fit_transform(features_v_unscaled[:,1:2]))
                if 'spacing_dev' in current_s_opts_v and features_v_unscaled[:,2:3].size>0:scols_v.append(get_s(current_s_opts_v['spacing_dev']).fit_transform(features_v_unscaled[:,2:3]))
                if not scols_v or len(scols_v)!=features_v_unscaled.shape[1]: scaled_fv=RobustScaler().fit_transform(features_v_unscaled) if features_v_unscaled.size>0 else features_v_unscaled
                else: scaled_fv=np.hstack(scols_v)
            except Exception as e: print(f"Err scaling valleys DBSCAN: {e}. Unscaled."); scaled_fv=features_v_unscaled

            if scaled_fv is not None and scaled_fv.shape[0]>=dbscan_min_samples_valleys:
                dbscan_v=DBSCAN(eps=dbscan_eps_valleys,min_samples=dbscan_min_samples_valleys);labels_v=dbscan_v.fit_predict(scaled_fv)
                core_lbl_v=labels_v[labels_v!=-1]
                if core_lbl_v.size>0:mc_lbl_v=Counter(core_lbl_v).most_common(1)[0][0];true_idx_v=np.where(labels_v==mc_lbl_v)[0];final_valleys_x_arr=rule_kept_valleys_x[true_idx_v];final_valleys_y_arr=rule_kept_valleys_y[true_idx_v]
                else:final_valleys_x_arr,final_valleys_y_arr=np.array([]),np.array([])
            elif scaled_fv is not None:
                 if verbose:print(f"    Too few valley samples for DBSCAN. Keeping rule-survivors.")
                 if scaled_fv.shape[0]>0: labels_v=-np.ones(scaled_fv.shape[0],dtype=int)
        dbscan_valleys_plot_data = (features_v_unscaled if 'features_v_unscaled' in locals() else None, scaled_fv, labels_v, np.copy(rule_kept_valleys_x), np.copy(rule_kept_valleys_y), -np.ones_like(rule_kept_valleys_x))
    else:
        if verbose: print(f"  Skipping DBSCAN for valleys (only {rule_kept_valleys_x.size} candidates or no peaks from DBSCAN).")


    # --- Final Alternation & Return ---
    if final_peaks_x_arr.size > 0 or final_valleys_x_arr.size > 0:
        fpxa,fpya,fvxa,fvya=_ensure_alternation((final_peaks_x_arr,final_peaks_y_arr),(final_valleys_x_arr,final_valleys_y_arr))
    else: fpxa,fpya,fvxa,fvya=np.array([]),np.array([]),np.array([]),np.array([])
    
    plot_data_for_orchestrator = (dbscan_peaks_plot_data, dbscan_valleys_plot_data)
    return fpxa,fpya,fvxa,fvya, plot_data_for_orchestrator

def filter_lfc_extrema_v8_final_rules_with_plots(
    peaks_x_cand, peaks_y_cand_refined,
    valleys_x_cand, valleys_y_cand_refined,
    original_x_axis, original_y_axis,
    # Parameters
    spacing_poly_degree=2, 
    spacing_uncertainty_const=2.0, 
    spacing_max_dev_factor=3.0,
    envelope_savgol_window=51, 
    envelope_savgol_polyorder=3,
    ratio_trend_savgol_window=101, 
    ratio_trend_savgol_polyorder=2,
    envelope_ratio_max_abs_dev_from_trend = 0.15,
    peak_snr_min_thresh_poisson=3.0,
    noise_estimation_window_pixels=15, 
    apply_dbscan_filter=False,
    global_fallback_noise_std=None,
    plot_filter_diagnostics=False, 
    verbose=False
):
    # --- 0. Initial Setup & Alternation ---
    if global_fallback_noise_std is None:
        global_fallback_noise_std = robust_noise_std(original_y_axis - np.median(original_y_axis))

    current_peaks_x, current_peaks_y, \
    current_valleys_x, current_valleys_y = _ensure_alternation(
        (peaks_x_cand, peaks_y_cand_refined),
        (valleys_x_cand, valleys_y_cand_refined)
    )
    if verbose: print(f"  RuleFilter v8 Input: {len(current_peaks_x)}P, {len(current_valleys_x)}V (Y refined)")

    plot_cand_peaks_x_alt = np.copy(current_peaks_x)
    plot_cand_peaks_y_alt = np.copy(current_peaks_y)
    plot_cand_valleys_x_alt = np.copy(current_valleys_x)
    plot_cand_valleys_y_alt = np.copy(current_valleys_y)

    if not current_peaks_x.size or not current_valleys_x.size:
        print("  Not enough peaks OR valleys after initial alternation for v8 filtering.")
        plot_data_ph = (None,None,None, np.array([]),np.array([]),np.array([]))
        return current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y, plot_data_ph

    # --- 1. Fit Global Spacing Models ---
    spacing_model_p, spacing_poly_trans_p = _fit_spacing_polynomial(current_peaks_x, poly_degree=spacing_poly_degree)
    spacing_model_v, spacing_poly_trans_v = _fit_spacing_polynomial(current_valleys_x, poly_degree=spacing_poly_degree)

    # --- 2. Fit Global Savgol Envelopes (on original_x_axis grid) ---
    peak_envelope = fit_envelope_savgol(current_peaks_x, current_peaks_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)
    valley_envelope = fit_envelope_savgol(current_valleys_x, current_valleys_y, original_x_axis, envelope_savgol_window, envelope_savgol_polyorder, verbose)
    if peak_envelope is None and verbose: print("  Warning: Failed to create peak envelope.")
    if valley_envelope is None and verbose: print("  Warning: Failed to create valley envelope.")
    
    # --- 3. Calculate and Smooth the V_env(x) / P_env(x) Ratio Trend ---
    R_exp_trend = None
    if peak_envelope is not None and valley_envelope is not None and \
       np.all(np.isfinite(peak_envelope)) and np.all(np.isfinite(valley_envelope)):
        P_env_for_ratio = np.maximum(peak_envelope, 1e-6 * (np.nanmax(peak_envelope) if np.any(np.isfinite(peak_envelope)) and np.nanmax(peak_envelope) > 0 else 1.0) )
        R_observed_globally = valley_envelope / P_env_for_ratio
        
        ratio_win = mathfunc.round_down_to_odd(int(ratio_trend_savgol_window))
        ratio_poly = int(ratio_trend_savgol_polyorder)
        if ratio_win <= ratio_poly: ratio_win = ratio_poly + (2 - ratio_poly % 2)
        
        if len(R_observed_globally) >= ratio_win:
            try: R_exp_trend = savgol_filter(R_observed_globally, window_length=ratio_win, polyorder=ratio_poly, mode="mirror")
            except ValueError: R_exp_trend = R_observed_globally 
        else: R_exp_trend = R_observed_globally
    if R_exp_trend is None and verbose: print("  Warning: Could not compute R_exp_trend for envelope ratio criterion.")

    # --- Data structures for storing metrics ---
    peak_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    valley_metrics = {'x':[],'y':[],'snr':[],'spacing_dev':[],'env_ratio_obs':[],'env_ratio_exp':[],'pass_snr':[],'pass_spacing':[],'pass_ratio':[],'pass_all_rules':[]}
    
    # --- Filter Peaks ---
    for i in range(len(current_peaks_x)):
        xp, yp = current_peaks_x[i], current_peaks_y[i]
        peak_metrics['x'].append(xp); peak_metrics['y'].append(yp)
        pass_pos, pass_snr_p, pass_spacing_p, pass_ratio_p = True, True, True, True
        if yp <= 1e-9: pass_pos = False
        current_snr_P_val, current_spacing_dev_p_val, obs_r_peak_val, exp_r_peak_val = np.nan, np.nan, np.nan, np.nan

        if pass_pos: # S/N for positive peaks
            
            # w_x = np.arange(xp - wsize, xp + wsize + 1, 1)
            # V_env_local = np.interp(w_x, original_x_axis, valley_envelope) if valley_envelope is not None else np.median(original_y_axis)
            # if not np.all(np.isfinite(V_env_local)): V_env_local = np.median(original_y_axis)
            # sum_V_env_local = np.sum(V_env_local)
            # sum_signal_local = np.sum(original_y_axis[ xp-wsize : xp+wsize ])
            # sum_signal_local = max(0, sum_signal_local)
            # noise_approx_P = np.sqrt(max(1, sum_signal_local) + max(1, sum_V_env_local)) 
            # y_local = original_y_axis[ xp-wsize : xp+wsize+1 ] - V_env_local
            # env_local = original_y_axis
            signal_P = yp - np.interp(xp, original_x_axis, valley_envelope) 
            signal_P = max(0,signal_P)
            
            # Define a window size over which to calculate S/N
            wsize = 15
            y_in_window = original_y_axis[xp - wsize : xp + wsize+1]
            mask        = mathfunc.sigclip1d_biased_low(y_in_window,
                                                        sigma_lower=3, 
                                                        sigma_upper=1.5,
                                                        plot=False)
            y_clipped   = y_in_window[mask]
            
            noise_approx_P = np.std(y_clipped)
            
            
            if noise_approx_P>1e-9:
                current_snr_P_val = signal_P/noise_approx_P
            else: current_snr_P_val = signal_P/1e-9
            # print(f'{xp:<4d}{signal_P=:>12.3f}\t{noise_approx_P=:>12.3f}{current_snr_P_val=:>8.3f}')
            if current_snr_P_val < peak_snr_min_thresh_poisson: pass_snr_p = False
        else: pass_snr_p = False
        peak_metrics['snr'].append(current_snr_P_val)

        if spacing_model_p: # Spacing
            devs=[];s_b_e,s_f_e=np.nan,np.nan
            if i>0:
                s_b_a=xp-current_peaks_x[i-1];
                xm_b=(xp+current_peaks_x[i-1])/2;
                s_b_e_arr=predict_spacing(xm_b,spacing_model_p,spacing_poly_trans_p)
                if s_b_e_arr is not None and np.isfinite(s_b_e_arr[0]):
                    s_b_e=s_b_e_arr[0];
                    devs.append(abs(s_b_a-s_b_e)/spacing_uncertainty_const)
            if i<len(current_peaks_x)-1:
                s_f_a=current_peaks_x[i+1]-xp;
                xm_f=(xp+current_peaks_x[i+1])/2;
                s_f_e_arr=predict_spacing(xm_f,spacing_model_p,spacing_poly_trans_p);
                if s_f_e_arr is not None and np.isfinite(s_f_e_arr[0]):
                    s_f_e=s_f_e_arr[0];
                    devs.append(abs(s_f_a-s_f_e)/spacing_uncertainty_const)
            if devs:current_spacing_dev_p_val=np.mean(devs)
            else:current_spacing_dev_p_val=0 
            if current_spacing_dev_p_val > spacing_max_dev_factor: pass_spacing_p=False
        peak_metrics['spacing_dev'].append(current_spacing_dev_p_val)
        
        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None: # Envelope Ratio
            P_env_val = np.interp(xp, original_x_axis, peak_envelope)
            V_env_val = np.interp(xp, original_x_axis, valley_envelope)
            exp_r_peak_val = np.interp(xp, original_x_axis, R_exp_trend)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_peak_val):
                if yp <= V_env_val : pass_ratio_p = False
                if pass_ratio_p and P_env_val > V_env_val and yp < (V_env_val + 0.3 * (P_env_val - V_env_val)): pass_ratio_p = False
                if pass_ratio_p and abs(P_env_val) > 1e-6:
                    obs_r_peak_val = V_env_val / P_env_val
                    if abs(obs_r_peak_val - exp_r_peak_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_p = False
                elif pass_ratio_p : pass_ratio_p = False
            else: pass_ratio_p = False
        else: pass_ratio_p = True # Cannot test
        peak_metrics['env_ratio_obs'].append(obs_r_peak_val); peak_metrics['env_ratio_exp'].append(exp_r_peak_val)
        peak_metrics['pass_snr'].append(pass_pos and pass_snr_p); peak_metrics['pass_spacing'].append(pass_spacing_p); peak_metrics['pass_ratio'].append(pass_ratio_p)
        peak_metrics['pass_all_rules'].append(pass_pos and pass_snr_p and pass_spacing_p and pass_ratio_p)

    # Filter Valleys (No S/N, uses R_exp_trend for ratio check)
    # ... (Valley filtering logic similar to peaks, but no S/N check. Populates valley_metrics) ...
    for i in range(len(current_valleys_x)):
        xv, yv = current_valleys_x[i], current_valleys_y[i]
        valley_metrics['x'].append(xv); valley_metrics['y'].append(yv)
        pass_spacing_v, pass_ratio_v = True, True
        valley_metrics['snr'].append(np.nan); valley_metrics['pass_snr'].append(True) # No S/N for valleys
        current_spacing_dev_v_val=np.nan; obs_r_valley_val=np.nan; exp_r_valley_val=np.nan
        if spacing_model_v: # Spacing logic for valleys
            devs_v=[];s_b_e_v,s_f_e_v=np.nan,np.nan
            if i>0:
                s_b_a=xv-current_valleys_x[i-1];
                xm_b=(xv+current_valleys_x[i-1])/2;
                s_e_arr=predict_spacing(xm_b,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_b_e_v=s_e_arr[0];
                    devs_v.append(abs(s_b_a-s_b_e_v)/spacing_uncertainty_const)
            if i<len(current_valleys_x)-1:
                s_f_a=current_valleys_x[i+1]-xv;
                xm_f=(xv+current_valleys_x[i+1])/2;
                s_e_arr=predict_spacing(xm_f,spacing_model_v,spacing_poly_trans_v);
                if s_e_arr is not None and np.isfinite(s_e_arr[0]):
                    s_f_e_v=s_e_arr[0];
                    devs_v.append(abs(s_f_a-s_f_e_v)/spacing_uncertainty_const)
            if devs_v:current_spacing_dev_v_val=np.mean(devs_v)
            else:current_spacing_dev_v_val=0
            if current_spacing_dev_v_val > spacing_max_dev_factor: pass_spacing_v=False
        valley_metrics['spacing_dev'].append(current_spacing_dev_v_val)
        if R_exp_trend is not None and peak_envelope is not None and valley_envelope is not None: # Envelope Ratio for Valleys
            P_env_val = np.interp(xv, original_x_axis, peak_envelope)
            V_env_val = np.interp(xv, original_x_axis, valley_envelope)
            exp_r_valley_val = np.interp(xv, original_x_axis, R_exp_trend)
            if np.isfinite(P_env_val) and np.isfinite(V_env_val) and np.isfinite(exp_r_valley_val):
                if yv >= P_env_val: pass_ratio_v = False
                if pass_ratio_v and P_env_val > V_env_val and yv > (V_env_val + 0.7*(P_env_val-V_env_val)): pass_ratio_v = False
                if pass_ratio_v and abs(P_env_val) > 1e-6:
                    obs_r_valley_val = V_env_val / P_env_val
                    if abs(obs_r_valley_val - exp_r_valley_val) > envelope_ratio_max_abs_dev_from_trend: pass_ratio_v = False
                elif pass_ratio_v: pass_ratio_v = False
            else: pass_ratio_v = False
        else: pass_ratio_v = True
        valley_metrics['env_ratio_obs'].append(obs_r_valley_val); valley_metrics['env_ratio_exp'].append(exp_r_valley_val if 'exp_r_valley_val' in locals() and np.isfinite(exp_r_valley_val) else np.nan)
        valley_metrics['pass_spacing'].append(pass_spacing_v); valley_metrics['pass_ratio'].append(pass_ratio_v)
        valley_metrics['pass_all_rules'].append(pass_spacing_v and pass_ratio_v) # Only spacing and ratio for valleys

    for key in peak_metrics: peak_metrics[key] = np.array(peak_metrics[key])
    for key in valley_metrics: valley_metrics[key] = np.array(valley_metrics[key])

    rule_kept_peaks_x = peak_metrics['x'][peak_metrics['pass_all_rules']]
    rule_kept_peaks_y = peak_metrics['y'][peak_metrics['pass_all_rules']]
    rule_kept_valleys_x = valley_metrics['x'][valley_metrics['pass_all_rules']]
    rule_kept_valleys_y = valley_metrics['y'][valley_metrics['pass_all_rules']]
    if verbose: print(f"  After rule-based pre-filter: {len(rule_kept_peaks_x)}P, {len(rule_kept_valleys_x)}V")

    # --- Diagnostic Plots for Rule-Based Filter ---
    if plot_filter_diagnostics:
        fig, axs = plt.subplots(4, 1, figsize=(15,16), sharex=True)
        fig.suptitle("Rule-Based Filter Diagnostics (Savgol Ratio Trend)", fontsize=16)
        xpr = original_x_axis

        # Panel 1: Data, Envelopes, Final Selection
        axs[0].plot(original_x_axis,original_y_axis,c='grey',alpha=0.4,label="Orig")
        if peak_envelope is not None:axs[0].plot(original_x_axis,peak_envelope,'b-',alpha=0.6,lw=1.5,label="P Env (Savgol)")
        if valley_envelope is not None:axs[0].plot(original_x_axis,valley_envelope,'g-',alpha=0.6,lw=1.5,label="V Env (Savgol)")
        axs[0].scatter(plot_cand_peaks_x_alt,plot_cand_peaks_y_alt,c='lightcoral',marker='^',s=20,alpha=0.5,label="P Cands")
        axs[0].scatter(plot_cand_valleys_x_alt,plot_cand_valleys_y_alt,c='lightgreen',marker='v',s=20,alpha=0.5,label="V Cands")
        if peak_metrics.get('x',np.array([])).size>0 and peak_metrics.get('pass_all_rules',np.array([])).size>0 and np.any(peak_metrics['pass_all_rules']):axs[0].scatter(peak_metrics['x'][peak_metrics['pass_all_rules']],peak_metrics['y'][peak_metrics['pass_all_rules']],c='blue',marker='^',s=40,label="Rule-Kept P")
        if valley_metrics.get('x',np.array([])).size>0 and valley_metrics.get('pass_all_rules',np.array([])).size>0 and np.any(valley_metrics['pass_all_rules']):axs[0].scatter(valley_metrics['x'][valley_metrics['pass_all_rules']],valley_metrics['y'][valley_metrics['pass_all_rules']],c='darkgreen',marker='v',s=40,label="Rule-Kept V")
        axs[0].set_title("Data, Envelopes, Rule-Filtered");axs[0].set_ylabel("Flux");axs[0].legend(fontsize='small',loc='upper right');axs[0].grid(True,ls=':')
        
        # Panel 2: S/N (Peaks only)
        if peak_metrics.get('x',np.array([])).size>0:
            axs[1].scatter(peak_metrics['x'][peak_metrics['pass_snr']],peak_metrics['snr'][peak_metrics['pass_snr']],c='b',marker='^',s=30,alpha=0.7,label="P S/N Pass")
            axs[1].scatter(peak_metrics['x'][~peak_metrics['pass_snr']],peak_metrics['snr'][~peak_metrics['pass_snr']],c='r',marker='x',s=50,label="P S/N Fail")
        axs[1].axhline(peak_snr_min_thresh_poisson,c='k',ls='--',label="P S/N Thresh");axs[1].set_title("Peak S/N ((yp - Venv) / noise_approx)");axs[1].set_ylabel("S/N");axs[1].legend(fontsize='small');axs[1].grid(True,ls=':');axs[1].set_yscale('symlog',linthresh=1,linscale=0.5)
        
        # Panel 3: Spacing Deviation
        if peak_metrics.get('x',np.array([])).size>0:axs[2].scatter(peak_metrics['x'][peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][peak_metrics['pass_spacing']],c='b',marker='^',s=30,alpha=0.7,label="P Spacing Pass");axs[2].scatter(peak_metrics['x'][~peak_metrics['pass_spacing']],peak_metrics['spacing_dev'][~peak_metrics['pass_spacing']],c='r',marker='x',s=50,label="P Spacing Fail")
        if valley_metrics.get('x',np.array([])).size>0:axs[2].scatter(valley_metrics['x'][valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][valley_metrics['pass_spacing']],c='g',marker='v',s=30,alpha=0.7,label="V Spacing Pass");axs[2].scatter(valley_metrics['x'][~valley_metrics['pass_spacing']],valley_metrics['spacing_dev'][~valley_metrics['pass_spacing']],c='m',marker='x',s=50,label="V Spacing Fail")
        axs[2].axhline(spacing_max_dev_factor,c='k',ls='--',label="Max Spacing Dev");axs[2].set_title("Norm Spacing Dev");axs[2].set_ylabel("Norm Spacing Dev");axs[2].legend(fontsize='small');axs[2].grid(True,ls=':');axs[2].set_yscale('log')
        
        # Panel 4: Envelope Ratio Deviation
        if R_exp_trend is not None and original_x_axis.size > 0: axs[3].plot(original_x_axis, R_exp_trend,'k:',alpha=0.8,label="Smoothed Venv/Penv Trend (R_exp)")
        if peak_metrics.get('x',np.array([])).size > 0 :
            valid_p_ratios = np.isfinite(peak_metrics['env_ratio_obs']) & np.isfinite(peak_metrics['env_ratio_exp'])
            if np.any(valid_p_ratios): # Only plot if there's valid data
                peak_ratio_devs = np.abs(peak_metrics['env_ratio_obs'][valid_p_ratios] - peak_metrics['env_ratio_exp'][valid_p_ratios])
                axs[3].scatter(peak_metrics['x'][valid_p_ratios][peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[peak_metrics['pass_ratio'][valid_p_ratios]], c='b', marker='^', s=30, alpha=0.7, label="P Ratio Pass")
                axs[3].scatter(peak_metrics['x'][valid_p_ratios][~peak_metrics['pass_ratio'][valid_p_ratios]], peak_ratio_devs[~peak_metrics['pass_ratio'][valid_p_ratios]],c='r', marker='x', s=50, label="P Ratio Fail")
        if valley_metrics.get('x',np.array([])).size > 0:
            valid_v_ratios = np.isfinite(valley_metrics['env_ratio_obs']) & np.isfinite(valley_metrics['env_ratio_exp'])
            if np.any(valid_v_ratios):
                valley_ratio_devs = np.abs(valley_metrics['env_ratio_obs'][valid_v_ratios] - valley_metrics['env_ratio_exp'][valid_v_ratios])
                axs[3].scatter(valley_metrics['x'][valid_v_ratios][valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[valley_metrics['pass_ratio'][valid_v_ratios]], c='g', marker='v', s=30, alpha=0.7, label="V Ratio Pass")
                axs[3].scatter(valley_metrics['x'][valid_v_ratios][~valley_metrics['pass_ratio'][valid_v_ratios]], valley_ratio_devs[~valley_metrics['pass_ratio'][valid_v_ratios]],c='m', marker='x', s=50, label="V Ratio Fail")
        axs[3].axhline(envelope_ratio_max_abs_dev_from_trend, c='k', ls='--', label="Max Ratio Dev")
        axs[3].set_title("Abs Dev from Smoothed Venv/Penv Ratio Trend");axs[3].set_ylabel("|Obs V/P - R_exp|");axs[3].legend(fontsize='small');axs[3].grid(True,ls=':');axs[3].set_xlabel("X-coord")
        
        plt.tight_layout(rect=[0,0,1,0.97]);plt.show()


    # --- DBSCAN Stage on Rule-Filtered Candidates ---
    # (This part is identical to filter_lfc_extrema_v7_hybrid_sep_dbscan, using the
    #  rule_kept_peaks_x/y and rule_kept_valleys_x/y as input to this stage)
    # ... (The full DBSCAN logic with _calculate_features_for_one_type, scaling, etc.) ...
    final_dbscan_peaks_x, final_dbscan_peaks_y = rule_kept_peaks_x, rule_kept_peaks_y
    final_dbscan_valleys_x, final_dbscan_valleys_y = rule_kept_valleys_x, rule_kept_valleys_y
    dbscan_peaks_plot_data=(None,None,None, np.copy(rule_kept_peaks_x), np.copy(rule_kept_peaks_y), np.ones_like(rule_kept_peaks_x))
    dbscan_valleys_plot_data=(None,None,None, np.copy(rule_kept_valleys_x), np.copy(rule_kept_valleys_y), -np.ones_like(rule_kept_valleys_x))

    if apply_dbscan_filter: # apply_dbscan_filter needs to be a parameter of this function
        # Separate DBSCAN for Peaks
        if rule_kept_peaks_x.size >= min_candidates_for_dbscan and rule_kept_valleys_x.size > 0:
            if verbose:print(f"  Applying DBSCAN to {len(rule_kept_peaks_x)} rule-filtered peaks...")
            features_p_unscaled=_calculate_features_for_one_type(rule_kept_peaks_x,rule_kept_peaks_y,rule_kept_valleys_x,rule_kept_valleys_y,True,original_y_axis,original_x_axis,noise_estimation_window_pixels,global_fallback_noise_std,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
            scaled_fp,labels_p=None,None
            if features_p_unscaled.shape[0]>0:
                current_s_opts_p = feature_scaling_options_peaks if feature_scaling_options_peaks is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
                def get_s(sname):
                    if sname=='minmax':return MinMaxScaler();
                    if sname=='standard':return StandardScaler();
                    if sname=='robust':return RobustScaler();
                    raise ValueError(f"Unk scaler: {sname}")
                scols_p=[]
                try:
                    if'x'in current_s_opts_p and features_p_unscaled[:,0:1].size>0:scols_p.append(get_s(current_s_opts_p['x']).fit_transform(features_p_unscaled[:,0:1]))
                    if'prom_snr'in current_s_opts_p and features_p_unscaled[:,1:2].size>0:scols_p.append(get_s(current_s_opts_p['prom_snr']).fit_transform(features_p_unscaled[:,1:2]))
                    if'spacing_dev'in current_s_opts_p and features_p_unscaled[:,2:3].size>0:scols_p.append(get_s(current_s_opts_p['spacing_dev']).fit_transform(features_p_unscaled[:,2:3]))
                    if not scols_p or len(scols_p)!=features_p_unscaled.shape[1]: scaled_fp=RobustScaler().fit_transform(features_p_unscaled) if features_p_unscaled.size>0 else features_p_unscaled
                    else: scaled_fp=np.hstack(scols_p)
                except Exception as e: print(f"Err scaling P DBSCAN:{e}.Unscaled."); scaled_fp=features_p_unscaled
                if scaled_fp is not None and scaled_fp.shape[0]>=dbscan_min_samples_peaks:
                    dbscan_p=DBSCAN(eps=dbscan_eps_peaks,min_samples=dbscan_min_samples_peaks);labels_p=dbscan_p.fit_predict(scaled_fp)
                    core_lbl_p=labels_p[labels_p!=-1]
                    if core_lbl_p.size>0:mc_lbl_p=Counter(core_lbl_p).most_common(1)[0][0];true_idx_p=np.where(labels_p==mc_lbl_p)[0];final_dbscan_peaks_x=rule_kept_peaks_x[true_idx_p];final_dbscan_peaks_y=rule_kept_peaks_y[true_idx_p]
                    else:final_dbscan_peaks_x,final_dbscan_peaks_y=np.array([]),np.array([])
                elif scaled_fp is not None:
                     if verbose:print(f"    Too few peak samples for DBSCAN. Keeping rule-survivors.")
                     if scaled_fp.shape[0]>0: labels_p=-np.ones(scaled_fp.shape[0],dtype=int)
            dbscan_peaks_plot_data=(features_p_unscaled if'features_p_unscaled'in locals()else None,scaled_fp,labels_p,np.copy(rule_kept_peaks_x),np.copy(rule_kept_peaks_y),np.ones_like(rule_kept_peaks_x))
        # Separate DBSCAN for Valleys
        if rule_kept_valleys_x.size>=min_candidates_for_dbscan and final_dbscan_peaks_x.size > 0: # Use DBSCAN-filtered peaks as opposites
            if verbose:print(f"  Applying DBSCAN to {len(rule_kept_valleys_x)} rule-filtered valleys...")
            features_v_unscaled=_calculate_features_for_one_type(rule_kept_valleys_x,rule_kept_valleys_y,final_dbscan_peaks_x,final_dbscan_peaks_y,False,original_y_axis,original_x_axis,noise_estimation_window_pixels,global_fallback_noise_std,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
            scaled_fv,labels_v=None,None
            if features_v_unscaled.shape[0]>0:
                current_s_opts_v = feature_scaling_options_valleys if feature_scaling_options_valleys is not None else {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
                scols_v=[]
                try:
                    if'x'in current_s_opts_v and features_v_unscaled[:,0:1].size>0:scols_v.append(get_s(current_s_opts_v['x']).fit_transform(features_v_unscaled[:,0:1]))
                    if'prom_snr'in current_s_opts_v and features_v_unscaled[:,1:2].size>0:scols_v.append(get_s(current_s_opts_v['prom_snr']).fit_transform(features_v_unscaled[:,1:2]))
                    if'spacing_dev'in current_s_opts_v and features_v_unscaled[:,2:3].size>0:scols_v.append(get_s(current_s_opts_v['spacing_dev']).fit_transform(features_v_unscaled[:,2:3]))
                    if not scols_v or len(scols_v)!=features_v_unscaled.shape[1]: scaled_fv=RobustScaler().fit_transform(features_v_unscaled) if features_v_unscaled.size>0 else features_v_unscaled
                    else: scaled_fv=np.hstack(scols_v)
                except Exception as e: print(f"Err scaling V DBSCAN:{e}.Unscaled."); scaled_fv=features_v_unscaled
                if scaled_fv is not None and scaled_fv.shape[0]>=dbscan_min_samples_valleys:
                    dbscan_v=DBSCAN(eps=dbscan_eps_valleys,min_samples=dbscan_min_samples_valleys);labels_v=dbscan_v.fit_predict(scaled_fv)
                    core_lbl_v=labels_v[labels_v!=-1]
                    if core_lbl_v.size>0:mc_lbl_v=Counter(core_lbl_v).most_common(1)[0][0];true_idx_v=np.where(labels_v==mc_lbl_v)[0];final_dbscan_valleys_x=rule_kept_valleys_x[true_idx_v];final_dbscan_valleys_y=rule_kept_valleys_y[true_idx_v]
                    else:final_dbscan_valleys_x,final_valleys_y_arr=np.array([]),np.array([])
                elif scaled_fv is not None:
                    if verbose:print(f"    Too few valley samples for DBSCAN. Keeping rule-survivors.")
                    if scaled_fv.shape[0]>0: labels_v=-np.ones(scaled_fv.shape[0],dtype=int)
            dbscan_valleys_plot_data = (features_v_unscaled if 'features_v_unscaled' in locals() else None, scaled_fv, labels_v, np.copy(rule_kept_valleys_x), np.copy(rule_kept_valleys_y), -np.ones_like(rule_kept_valleys_x))
    else:
        if verbose: print("  DBSCAN filter stage was turned off. Using rule-based results.")
        # plot_data_for_orchestrator should reflect candidates that *would have* gone to DBSCAN
        # These are rule_kept_peaks/valleys combined.
        # Handled by initialization of dbscan_peaks/valleys_plot_info

    # --- Final Alternation & Return ---
    # if final_dbscan_peaks_x.size > 0 or final_dbscan_valleys_x.size > 0: # Use final_dbscan arrays here
    #     fpxa,fpya,fvxa,fvya=_ensure_alternation(
    #         (final_dbscan_peaks_x,final_dbscan_peaks_y),
    #         (final_dbscan_valleys_x,final_dbscan_valleys_y)
    #     )
    # else: fpxa,fpya,fvxa,fvya=np.array([]),np.array([]),np.array([]),np.array([])
    fpxa,fpya,fvxa,fvya = final_dbscan_peaks_x,final_dbscan_peaks_y, \
                          final_dbscan_valleys_x,final_dbscan_valleys_y
    plot_data_for_orchestrator_final = (dbscan_peaks_plot_data, dbscan_valleys_plot_data)
    return fpxa,fpya,fvxa,fvya, plot_data_for_orchestrator_final


def apply_clustering_filter_to_extrema_v3(
    initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
    dbscan_eps, dbscan_min_samples, min_features_for_clustering,
    original_y_data_for_noise, original_x_data_for_noise,
    global_noise_std_for_fallback, noise_estimation_window_pixels_feat,
    spacing_poly_degree_feat, spacing_uncertainty_const_feat,
    scaling_options: dict = None
):
    _, _, _, _, \
    all_cand_x, all_cand_y, all_cand_types = _ensure_alternation_and_tag( # This specific version is needed here
        initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y
    )
    if all_cand_x.size < min_features_for_clustering:
        print(f"    Not enough combined cands ({all_cand_x.size}) for clust. Ret all alt cands.")
        fp_x=all_cand_x[all_cand_types==1];fp_y=all_cand_y[all_cand_types==1];fv_x=all_cand_x[all_cand_types==-1];fv_y=all_cand_y[all_cand_types==-1]
        return (fp_x,fp_y,fv_x,fv_y, (None,None,all_cand_x,all_cand_y,all_cand_types))

    features_unscaled = _calculate_features_for_extrema_v3(all_cand_x,all_cand_y,all_cand_types,original_y_data_for_noise,original_x_data_for_noise,noise_estimation_window_pixels_feat,global_noise_std_for_fallback,spacing_poly_degree_feat,spacing_uncertainty_const_feat)
    
    scaled_features_combined=None; labels=None
    cl_x,cl_y,cl_t = all_cand_x,all_cand_y,all_cand_types # Default if no scaling/clustering

    if features_unscaled.shape[0]>0 and features_unscaled.shape[1]==3:
        if scaling_options is None: scaling_options={'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
        def get_s(sname):
            if sname=='minmax':return MinMaxScaler()
            if sname=='standard':return StandardScaler()
            if sname=='robust':return RobustScaler();
            raise ValueError(f"Unk scaler: {sname}")
        scols=[]
        try: # Wrap scaling in try-except for robustness
            if 'x' in scaling_options and features_unscaled[:,0:1].size > 0: scols.append(get_s(scaling_options['x']).fit_transform(features_unscaled[:,0:1]))
            if 'prom_snr' in scaling_options and features_unscaled[:,1:2].size > 0: scols.append(get_s(scaling_options['prom_snr']).fit_transform(features_unscaled[:,1:2]))
            if 'spacing_dev' in scaling_options and features_unscaled[:,2:3].size > 0: scols.append(get_s(scaling_options['spacing_dev']).fit_transform(features_unscaled[:,2:3]))
            
            if not scols or len(scols) != features_unscaled.shape[1] :
                 if features_unscaled.size > 0: scaled_features_combined=RobustScaler().fit_transform(features_unscaled)
                 else: scaled_features_combined = features_unscaled
            else: scaled_features_combined=np.hstack(scols)
        except Exception as e:
            print(f"Error during feature scaling: {e}. Using unscaled features for DBSCAN (not recommended).")
            scaled_features_combined = features_unscaled # Fallback to unscaled


        if scaled_features_combined is not None and scaled_features_combined.shape[0] >= dbscan_min_samples : # DBSCAN needs enough samples
            dbscan=DBSCAN(eps=dbscan_eps,min_samples=dbscan_min_samples); labels=dbscan.fit_predict(scaled_features_combined)
            core_lbls=labels[labels!=-1]
            if core_lbls.size>0:
                mc_lbl=Counter(core_lbls).most_common(1)[0][0]; true_idx=np.where(labels==mc_lbl)[0]
                cl_x=all_cand_x[true_idx];cl_y=all_cand_y[true_idx];cl_t=all_cand_types[true_idx]
            else: cl_x,cl_y,cl_t=np.array([]),np.array([]),np.array([])
        elif scaled_features_combined is not None and scaled_features_combined.shape[0] > 0: # Not enough for DBSCAN, keep all
            print(f"    Too few samples ({scaled_features_combined.shape[0]}) for DBSCAN with min_samples={dbscan_min_samples}. Keeping all candidates.")
            cl_x,cl_y,cl_t = all_cand_x, all_cand_y, all_cand_types
        # else: cl_x,cl_y,cl_t are already defaulted to all_cand if features were empty

    fp_x=cl_x[cl_t==1];fp_y=cl_y[cl_t==1];fv_x=cl_x[cl_t==-1];fv_y=cl_y[cl_t==-1]
    # Use standard _ensure_alternation that takes (px,py), (vx,vy) tuples
    final_peaks_x_alt, final_peaks_y_alt, \
    final_valleys_x_alt, final_valleys_y_alt = _ensure_alternation( # THIS IS THE STANDARD VERSION
        (fp_x, fp_y), (fv_x, fv_y) # Pass as tuples
    )
    features_unscaled_for_plot = locals().get('features_unscaled', None)
    scaled_features_for_plot = locals().get('scaled_features_combined', None)
    labels_for_plot = locals().get('labels', None)
    
    pdata_comb=(features_unscaled_for_plot,
                scaled_features_for_plot,
                labels_for_plot,
                np.copy(all_cand_x), # Candidates that features were based on
                np.copy(all_cand_y),
                np.copy(all_cand_types))
                
    return (final_peaks_x_alt, final_peaks_y_alt, final_valleys_x_alt, final_valleys_y_alt,
            pdata_comb)

def apply_clustering_filter_to_extrema_v4( # Renamed to v4 for clarity of this change
    initial_peaks_x, initial_peaks_y_refined_orig, # Y-values are already refined
    initial_valleys_x, initial_valleys_y_refined_orig,
    dbscan_eps, dbscan_min_samples, min_features_for_clustering,
    original_y_data_for_noise, original_x_data_for_noise, # For S/N calculation context
    global_noise_std_for_fallback, noise_estimation_window_pixels_feat,
    spacing_poly_degree_feat, spacing_uncertainty_const_feat,
    scaling_options: dict = None
):
    # 1. Combine and strictly alternate (using the refined y-values)
    _, _, _, _, \
    all_cand_x, all_cand_y_refined, all_cand_types = _ensure_alternation_and_tag(
        initial_peaks_x, initial_peaks_y_refined_orig,
        initial_valleys_x, initial_valleys_y_refined_orig
    )

    if all_cand_x.size < min_features_for_clustering:
        print(f"    Not enough combined cands ({all_cand_x.size}) for clust. Ret all alt cands.")
        fp_x=all_cand_x[all_cand_types==1]; fp_y=all_cand_y_refined[all_cand_types==1] # Use refined y
        fv_x=all_cand_x[all_cand_types==-1]; fv_y=all_cand_y_refined[all_cand_types==-1] # Use refined y
        return (fp_x,fp_y,fv_x,fv_y, (None,None,all_cand_x,all_cand_y_refined,all_cand_types))

    # 2. Calculate features using _v3 (which now takes refined y for prom/depth calculation)
    features_unscaled = _calculate_features_for_extrema_v3(
        all_cand_x, all_cand_y_refined, all_cand_types, # Pass refined Y for prom/depth calc
        original_y_data_for_noise, original_x_data_for_noise,
        noise_estimation_window_pixels_feat, global_noise_std_for_fallback,
        spacing_poly_degree_feat, spacing_uncertainty_const_feat
    )
    
    # ... (Rest of scaling and DBSCAN logic from apply_clustering_filter_to_extrema_v3 is identical) ...
    scaled_features_combined=None; labels=None; cl_x,cl_y_refined,cl_t=all_cand_x,all_cand_y_refined,all_cand_types
    if features_unscaled.shape[0]>0 and features_unscaled.shape[1]==3:
        # ... (scaling logic identical to v3) ...
        if scaling_options is None: scaling_options={'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}
        def get_s(sname):
            if sname=='minmax':return MinMaxScaler();
            if sname=='standard':return StandardScaler();
            if sname=='robust':return RobustScaler();
            raise ValueError(f"Unk scaler: {sname}")
        scols=[]
        try:
            if 'x' in scaling_options and features_unscaled[:,0:1].size>0:scols.append(get_s(scaling_options['x']).fit_transform(features_unscaled[:,0:1]))
            if 'prom_snr' in scaling_options and features_unscaled[:,1:2].size>0:scols.append(get_s(scaling_options['prom_snr']).fit_transform(features_unscaled[:,1:2]))
            if 'spacing_dev' in scaling_options and features_unscaled[:,2:3].size>0:scols.append(get_s(scaling_options['spacing_dev']).fit_transform(features_unscaled[:,2:3]))
            if not scols or len(scols)!=features_unscaled.shape[1]:
                if features_unscaled.size>0:scaled_features_combined=RobustScaler().fit_transform(features_unscaled)
                else: scaled_features_combined=features_unscaled
            else: scaled_features_combined=np.hstack(scols)
        except Exception as e: print(f"Err scaling: {e}. Using unscaled."); scaled_features_combined=features_unscaled

        if scaled_features_combined is not None and scaled_features_combined.shape[0] >= dbscan_min_samples:
            dbscan=DBSCAN(eps=dbscan_eps,min_samples=dbscan_min_samples);labels=dbscan.fit_predict(scaled_features_combined)
            core_lbls=labels[labels!=-1]
            if core_lbls.size>0:
                mc_lbl=Counter(core_lbls).most_common(1)[0][0];true_idx=np.where(labels==mc_lbl)[0]
                cl_x=all_cand_x[true_idx];cl_y_refined=all_cand_y_refined[true_idx];cl_t=all_cand_types[true_idx]
            else: cl_x,cl_y_refined,cl_t=np.array([]),np.array([]),np.array([])
        elif scaled_features_combined is not None and scaled_features_combined.shape[0]>0:
            print(f"    Too few samples ({scaled_features_combined.shape[0]}) for DBSCAN. Keeping all cands.")
            cl_x,cl_y_refined,cl_t=all_cand_x,all_cand_y_refined,all_cand_types
            labels = -np.ones(len(cl_x), dtype=int) # Mark all as outliers if not clustered
            
    # Separate based on clustered types, Y values are already refined
    fp_x=cl_x[cl_t==1]; fp_y=cl_y_refined[cl_t==1]
    fv_x=cl_x[cl_t==-1]; fv_y=cl_y_refined[cl_t==-1]
    
    # Final alternation on results that already have refined Y
    fpa_x,fpa_y,fva_x,fva_y = _ensure_alternation((fp_x,fp_y),(fv_x,fv_y))
    
    pdata_comb=(features_unscaled if 'features_unscaled' in locals() else None,
                scaled_features_combined, labels,
                np.copy(all_cand_x),np.copy(all_cand_y_refined),np.copy(all_cand_types)) # Store refined y with candidates
    return (fpa_x,fpa_y,fva_x,fva_y,pdata_comb)

def detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed,
                                      deriv_method='coeff'):
    """
    Detects initial peak and valley candidates from smoothed data using derivatives.
    """
    if not isinstance(y_smoothed, np.ndarray): y_smoothed = np.asarray(y_smoothed)
    if not isinstance(x_coords_for_smoothed, np.ndarray): x_coords_for_smoothed = np.asarray(x_coords_for_smoothed)

    if len(y_smoothed) < 3: # Need at least 3 points for derivatives
        print("Warning in detect_initial_extrema_candidates: Smoothed data too short.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    derivative1st = mathfunc.derivative1d(y_smoothed, x=None, order=1, method=deriv_method)
    derivative2nd = mathfunc.derivative1d(y_smoothed, x=None, order=2, method=deriv_method)

    if derivative1st.size < 2: # Need at least 2 points for np.diff
        print("Warning in detect_initial_extrema_candidates: First derivative too short for sign change detection.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Threshold for first derivative being "close to zero" - currently used for plotting, not direct candidate selection here
    # deriv1_thresh = first_deriv_zero_threshold_factor * np.max(np.abs(derivative1st)) if derivative1st.size > 0 else 1e-9
    
    candidate_indices = []
    sign_changes = np.diff(np.sign(derivative1st))
    crossings_idx = np.where(np.abs(sign_changes) == 2)[0] # Indices *before* the sign change

    for idx in crossings_idx:
        # Refine to point with derivative1st closer to zero
        if idx + 1 < derivative1st.size: # Ensure idx+1 is a valid index
            if np.abs(derivative1st[idx]) < np.abs(derivative1st[idx + 1]):
                candidate_indices.append(idx)
            else:
                candidate_indices.append(idx + 1)
        else: # If idx is the second to last point, only idx can be the extremum from this crossing
            candidate_indices.append(idx)
            
    candidate_indices = sorted(list(set(candidate_indices))) # Unique sorted indices
    candidate_indices = np.array(candidate_indices, dtype=int)

    if candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Filter by 2nd derivative sign & ensure indices are valid
    valid_candidate_indices = candidate_indices[
        (candidate_indices < len(derivative2nd)) &
        (candidate_indices < len(x_coords_for_smoothed)) & # Also check against x_coords length
        (candidate_indices < len(y_smoothed)) # And y_smoothed length
    ]
    
    if valid_candidate_indices.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    max_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] < 0]
    min_ind = valid_candidate_indices[derivative2nd[valid_candidate_indices] > 0]

    peaks_x = x_coords_for_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    peaks_y = y_smoothed[max_ind] if max_ind.size > 0 else np.array([])
    valleys_x = x_coords_for_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    valleys_y = y_smoothed[min_ind] if min_ind.size > 0 else np.array([])
    
    return peaks_x, peaks_y, valleys_x, valleys_y

# --- SECTION 3: REFINEMENT TO ORIGINAL DATA (Parabolic/Weighted Mean Y) ---
def refine_extrema_to_original_pixel_grid( # Renamed for clarity
    extrema_x_from_smoothed, # X-positions from smoothed data (physical scale)
    original_x_axis,         # Original physical x-coordinates of the spectrum
    original_y_axis,         # Original y-flux values of the spectrum
    y_refinement_method='weighted_mean', # 'direct', 'weighted_mean', 'parabolic_vertex', 'parabolic_at_barycenter'
    y_refinement_weights=None  # For 'weighted_mean', e.g., [0.25, 0.5, 0.25]
):
    """
    Refines extrema by:
    1. X-coordinate: Finding the *integer pixel index* in original_x_axis closest
       to each x_from_smoothed.
    2. Y-coordinate: Calculating a refined y-value from original_y_axis centered
       at this integer pixel index, using the specified y_refinement_method.

    Args:
        extrema_x_from_smoothed (np.array): X-coords of extrema from smoothed data (physical scale).
        original_x_axis (np.array): The original physical x-coordinates of the spectrum (must be sorted).
        original_y_axis (np.array): The original y-flux values of the spectrum.
        y_refinement_method (str): Method for y-value refinement.
        y_refinement_weights (list/np.array, optional): Weights for 'weighted_mean'.

    Returns:
        tuple: (refined_x_indices, refined_y_values)
               refined_x_indices: NumPy array of integer pixel indices on the original grid.
               refined_y_values: NumPy array of corresponding refined y-values from original_y_axis.
    """
    original_x_axis = np.asarray(original_x_axis)
    original_y_axis = np.asarray(original_y_axis)
    extrema_x_input = np.asarray(extrema_x_from_smoothed) # Ensure it's an array

    if not original_x_axis.size or not original_y_axis.size or not extrema_x_input.size:
        # Return empty arrays of correct types if no input or no data to refine
        return np.array([], dtype=int), np.array([], dtype=float)

    # Prepare output arrays
    refined_x_indices_list = []
    refined_y_values_list = []

    if y_refinement_method == 'weighted_mean':
        if y_refinement_weights is None:
            y_refinement_weights = np.array([1/3., 1/3., 1/3.]) # Simple average
        else:
            y_refinement_weights = np.asarray(y_refinement_weights)
            if len(y_refinement_weights) != 3:
                print("Warning: y_refinement_weights must be length 3 for 'weighted_mean'. Using simple average.")
                y_refinement_weights = np.array([1/3., 1/3., 1/3.])
            elif not np.isclose(np.sum(y_refinement_weights), 1.0):
                print("Warning: y_refinement_weights do not sum to 1. Normalizing.")
                y_refinement_weights = y_refinement_weights / np.sum(y_refinement_weights)
    
    for x_val_smoothed in extrema_x_input:
        # Find index of closest point in original_x_axis
        # np.searchsorted finds insertion point.
        # If original_x_axis itself represents pixel indices (0, 1, 2,...),
        # then rounding x_val_smoothed (if it was supersampled index) and clamping
        # might be an alternative. But since original_x_axis can be physical units,
        # finding the closest point is more general.
        idx_map = np.searchsorted(original_x_axis, x_val_smoothed, side='left')
        
        center_idx_orig = 0 # Default for safety
        if idx_map == 0:
            center_idx_orig = 0
        elif idx_map == len(original_x_axis):
            center_idx_orig = len(original_x_axis) - 1
        else:
            # Compare distance to original_x_axis[idx_map-1] and original_x_axis[idx_map]
            if abs(original_x_axis[idx_map-1] - x_val_smoothed) < abs(original_x_axis[idx_map] - x_val_smoothed):
                center_idx_orig = idx_map - 1
            else:
                center_idx_orig = idx_map
        
        refined_x_indices_list.append(center_idx_orig) # Store the integer pixel index

        # --- Get refined Y value based on method, centered at center_idx_orig ---
        y_c_val_orig = original_y_axis[center_idx_orig] # Value at the chosen original pixel
        
        refined_y = y_c_val_orig # Default to direct value

        if y_refinement_method != 'direct':
            idx_l, idx_r = center_idx_orig - 1, center_idx_orig + 1
            if idx_l >= 0 and idx_r < len(original_y_axis): # Enough points for 3-point methods
                y_l_val_orig = original_y_axis[idx_l]
                y_r_val_orig = original_y_axis[idx_r]

                if y_refinement_method == 'weighted_mean':
                    refined_y = (y_refinement_weights[0] * y_l_val_orig +
                                 y_refinement_weights[1] * y_c_val_orig +
                                 y_refinement_weights[2] * y_r_val_orig)
                
                elif y_refinement_method == 'parabolic_vertex' or \
                     y_refinement_method == 'parabolic_at_barycenter':
                    
                    # Parabola: y_rel = a * x_rel^2 + b * x_rel + c, for x_rel = -1, 0, 1
                    # Coefficients relative to center_idx_orig as x_rel=0
                    c_parab = y_c_val_orig
                    b_parab = (y_r_val_orig - y_l_val_orig) / 2.0
                    a_parab = (y_l_val_orig + y_r_val_orig - 2.0 * y_c_val_orig) / 2.0

                    if abs(a_parab) < 1e-9: # Denominator 2a for vertex is near zero (linear/flat)
                        refined_y = y_c_val_orig
                    else:
                        if y_refinement_method == 'parabolic_vertex':
                            # y_vertex = c - b^2 / (4a) but using our a,b,c: a0 - a1^2/(4*a2)
                            y_vertex = c_parab - (b_parab**2) / (4 * a_parab)
                            refined_y = np.clip(y_vertex, 
                                                min(y_l_val_orig, y_c_val_orig, y_r_val_orig),
                                                max(y_l_val_orig, y_c_val_orig, y_r_val_orig))
                        
                        elif y_refinement_method == 'parabolic_at_barycenter':
                            flux_offset = 0.0
                            min_local_flux = min(y_l_val_orig, y_c_val_orig, y_r_val_orig)
                            if min_local_flux <= 0:
                                flux_offset = abs(min_local_flux) + 1e-6 # Ensure positive weights
                            
                            yl_w = y_l_val_orig + flux_offset
                            yc_w = y_c_val_orig + flux_offset
                            yr_w = y_r_val_orig + flux_offset
                            sum_fluxes = yl_w + yc_w + yr_w
                            
                            if abs(sum_fluxes) < 1e-9: x_bary_shift = 0.0
                            else: x_bary_shift = (yl_w * (-1) + yr_w * (1)) / sum_fluxes # Relative to center pixel
                            
                            y_at_bary = c_parab + b_parab * x_bary_shift + a_parab * (x_bary_shift**2)
                            refined_y = np.clip(y_at_bary,
                                                min(y_l_val_orig, y_c_val_orig, y_r_val_orig),
                                                max(y_l_val_orig, y_c_val_orig, y_r_val_orig))
            # else: refined_y remains y_c_val_orig (not enough points for 3-point window)
        
        refined_y_values_list.append(refined_y)

    return np.array(refined_x_indices_list, dtype=int), np.array(refined_y_values_list, dtype=float)

def refine_extrema_to_original_data(
    detected_peaks_xy_tuple,
    detected_valleys_xy_tuple,
    original_x_axis,
    original_y_axis,
    y_refinement_method='parabolic_vertex', # 'direct', 'weighted_mean', 'flux_weighted_mean', 'parabolic_vertex', 'parabolic_at_barycenter'
    y_refinement_weights=None # For 'weighted_mean', e.g., [0.25, 0.5, 0.25]
):
    """
    Refines extrema:
    1. X-coordinate: Maps to the *nearest* x-pixel in original_x_axis.
    2. Y-coordinate: Calculated from original_y_axis around the mapped x-pixel using a specified method.
    """
    original_x_axis = np.asarray(original_x_axis)
    original_y_axis = np.asarray(original_y_axis)
    
    if y_refinement_method == 'weighted_mean':
        if y_refinement_weights is None:
            y_refinement_weights = np.array([1/3., 1/3., 1/3.]) # Simple average
        else:
            y_refinement_weights = np.asarray(y_refinement_weights)
            if len(y_refinement_weights) != 3:
                print("Warning: y_refinement_weights must be length 3. Using simple average.")
                y_refinement_weights = np.array([1/3., 1/3., 1/3.])
            elif not np.isclose(np.sum(y_refinement_weights), 1.0):
                print("Warning: y_refinement_weights do not sum to 1. Normalizing.")
                y_refinement_weights = y_refinement_weights / np.sum(y_refinement_weights)

    refined_peaks_list = []
    refined_valleys_list = []

    if not original_x_axis.size or not original_y_axis.size:
        return (np.array([]), np.array([])), (np.array([]), np.array([]))

    def get_refined_y_value(center_idx_orig, y_array_orig, method, weights, is_peak): # Added is_peak
        # Central pixel value is the fallback
        y_c_val = y_array_orig[center_idx_orig]

        # Ensure indices are within bounds for 3-point window
        idx_l, idx_c, idx_r = center_idx_orig - 1, center_idx_orig, center_idx_orig + 1
        if idx_l < 0 or idx_r >= len(y_array_orig): # Not enough points for 3-point methods
            return y_c_val

        y_l_val, y_r_val = y_array_orig[idx_l], y_array_orig[idx_r]

        if method == 'direct':
            return y_c_val
        
        elif method == 'weighted_mean': # Fixed weights
            return weights[0]*y_l_val + weights[1]*y_c_val + weights[2]*y_r_val
        
        elif method == 'flux_weighted_mean': # Fluxes as weights (Poisson error motivated for centroid)
            # Ensure weights are positive and sum is not zero
            # Use absolute values for weights if fluxes can be negative (e.g. after baseline subtraction)
            # or add a small positive constant to all fluxes before weighting if they can be zero.
            # For simplicity, assuming y_l_val, y_c_val, y_r_val are > 0 for weighting.
            # If not, this method needs more robust handling (e.g. use y_i - min(y_window) + epsilon as weights)
            
            # For peaks, we want to weight by actual flux. For valleys, by "inverted" flux,
            # or more simply, still use flux but expect the barycenter to be near the min.
            # Let's use raw flux values as weights for simplicity of barycenter.
            # A small epsilon to prevent division by zero if all fluxes are zero
            sum_fluxes = y_l_val + y_c_val + y_r_val + 1e-9 
            # This calculates the flux-weighted average y-value over the 3 pixels, not a barycenter position.
            # If the goal is a flux-weighted y-value, this is it:
            # y_refined = (y_l_val*y_l_val + y_c_val*y_c_val + y_r_val*y_r_val) / sum_fluxes
            # This interpretation seems off. Let's stick to fixed weights or parabola for now.
            # If "Poissonian error weighted mean" for y_value estimation: weights are 1/sigma^2 ~ 1/y
            # This means smaller y get HIGHER weight, which pulls peak down, valley up.
            # This is standard for fitting data points, but maybe not for estimating peak/valley y-value itself.
            # Reverting to simpler methods or parabolic for now unless specific formula is given.
            # For now, 'flux_weighted_mean' will be a simple 3-point average.
            print("Note: 'flux_weighted_mean' currently implemented as simple 3-point average.")
            return (y_l_val + y_c_val + y_r_val) / 3.0

        elif method == 'parabolic_vertex' or method == 'parabolic_at_barycenter':
            # Parabola: y_rel = a * x_rel^2 + b * x_rel + c
            # x_rel = -1, 0, 1
            # y_points = [y_l_val, y_c_val, y_r_val]
            # c = y_c_val
            # a = (y_l_val + y_r_val)/2 - y_c_val
            # b = (y_r_val - y_l_val)/2
            
            # Using notation from https://sgolay.hut भा.uk/onlinecalc.htm (coeff order a0, a1, a2)
            # y = a0 + a1*x + a2*x^2
            # For x = -1, 0, 1:
            # a0 = y_c_val
            # a1 = (y_r_val - y_l_val) / 2
            # a2 = (y_l_val + y_r_val - 2*y_c_val) / 2
            
            a0 = y_c_val
            a1 = (y_r_val - y_l_val) / 2.0
            a2 = (y_l_val + y_r_val - 2.0 * y_c_val) / 2.0

            if abs(a2) < 1e-9: # Denominator for vertex is 2*a2. If linear or flat, return central value.
                return y_c_val 
            
            if method == 'parabolic_vertex':
                # Sub-pixel x-shift of vertex from center pixel: x_v_shift = -a1 / (2*a2)
                # Y-value at vertex: y_vertex = a0 + a1*x_v_shift + a2*x_v_shift^2
                # OR y_vertex = a0 - a1^2 / (4*a2)
                y_vertex = a0 - (a1**2) / (4 * a2)
                
                # Clamp to avoid extrapolation far beyond observed in noisy cases
                min_observed = min(y_l_val, y_c_val, y_r_val)
                max_observed = max(y_l_val, y_c_val, y_r_val)
                return np.clip(y_vertex, min_observed, max_observed)

            elif method == 'parabolic_at_barycenter':
                # Flux barycenter shift (relative to center pixel 0)
                # Add small positive offset to fluxes if they can be zero or negative to ensure positive weights
                flux_offset = 0.0
                if min(y_l_val, y_c_val, y_r_val) <=0:
                    flux_offset = abs(min(y_l_val, y_c_val, y_r_val)) + 1e-6 # Ensure positive
                
                yl_w = y_l_val + flux_offset
                yc_w = y_c_val + flux_offset
                yr_w = y_r_val + flux_offset
                sum_fluxes = yl_w + yc_w + yr_w
                
                if abs(sum_fluxes) < 1e-9: return y_c_val # All fluxes near zero

                x_barycenter_shift = (yl_w * (-1) + yc_w * (0) + yr_w * (1)) / sum_fluxes
                
                # Evaluate parabola at this barycenter shift
                y_at_barycenter = a0 + a1 * x_barycenter_shift + a2 * (x_barycenter_shift**2)
                
                min_observed = min(y_l_val, y_c_val, y_r_val)
                max_observed = max(y_l_val, y_c_val, y_r_val)
                return np.clip(y_at_barycenter, min_observed, max_observed)
        else:
            print(f"Warning: Unknown y_refinement_method '{method}'. Using direct value.")
            return y_c_val

    # --- Refine Peaks ---
    detected_peaks_x = np.asarray(detected_peaks_xy_tuple[0]) # X values from smoothed data
    if detected_peaks_x.size > 0:
        insertion_indices = np.searchsorted(original_x_axis, detected_peaks_x, side='left')
        for i, det_peak_x_val in enumerate(detected_peaks_x):
            idx = insertion_indices[i]
            if idx == 0: closest_orig_idx = 0
            elif idx == len(original_x_axis): closest_orig_idx = len(original_x_axis) - 1
            else:
                closest_orig_idx = idx - 1 if abs(original_x_axis[idx-1] - det_peak_x_val) < abs(original_x_axis[idx] - det_peak_x_val) else idx
            
            # X-position is fixed to original_x_axis[closest_orig_idx]
            refined_x_val = original_x_axis[closest_orig_idx]
            # Y-value refinement happens around this closest_orig_idx
            refined_y_val = get_refined_y_value(closest_orig_idx, original_y_axis, 
                                                y_refinement_method, y_refinement_weights, is_peak=True)
            refined_peaks_list.append((refined_x_val, refined_y_val))

    # --- Refine Valleys ---
    detected_valleys_x = np.asarray(detected_valleys_xy_tuple[0]) # X values from smoothed data
    if detected_valleys_x.size > 0:
        insertion_indices = np.searchsorted(original_x_axis, detected_valleys_x, side='left')
        for i, det_valley_x_val in enumerate(detected_valleys_x):
            idx = insertion_indices[i]
            if idx == 0: closest_orig_idx = 0
            elif idx == len(original_x_axis): closest_orig_idx = len(original_x_axis) - 1
            else:
                closest_orig_idx = idx - 1 if abs(original_x_axis[idx-1] - det_valley_x_val) < abs(original_x_axis[idx] - det_valley_x_val) else idx

            refined_x_val = original_x_axis[closest_orig_idx]
            refined_y_val = get_refined_y_value(closest_orig_idx, original_y_axis,
                                                y_refinement_method, y_refinement_weights, is_peak=False)
            refined_valleys_list.append((refined_x_val, refined_y_val))

    # --- Remove duplicates based on X and sort ---
    # (Using dictionary ensures unique X, keeping first encountered after sorting)
    unique_refined_peaks_dict = {}
    if refined_peaks_list:
        refined_peaks_list.sort(key=lambda p: (p[0], -p[1])) # Sort by X, then Y desc for peaks
        for x,y_val in refined_peaks_list:
            # Use np.isclose for comparing x if they are float and very close
            # For simplicity, exact match after rounding, or simply take first if sorted well
            # To handle very close x mapping to same original x, dict keying on original x is robust
            # Here, refined_x_val IS original_x_axis[closest_orig_idx], so it's already on grid
            if x not in unique_refined_peaks_dict:
                 unique_refined_peaks_dict[x] = y_val
    final_refined_peaks = sorted(list(unique_refined_peaks_dict.items()))

    unique_refined_valleys_dict = {}
    if refined_valleys_list:
        refined_valleys_list.sort(key=lambda p: (p[0], p[1])) # Sort by X, then Y asc for valleys
        for x,y_val in refined_valleys_list:
            if x not in unique_refined_valleys_dict:
                 unique_refined_valleys_dict[x] = y_val
    final_refined_valleys = sorted(list(unique_refined_valleys_dict.items()))

    rpx,rpy = (np.array(list(zip(*final_refined_peaks))[0]), np.array(list(zip(*final_refined_peaks))[1])) if final_refined_peaks else (np.array([]),np.array([]))
    rvx,rvy = (np.array(list(zip(*final_refined_valleys))[0]), np.array(list(zip(*final_refined_valleys))[1])) if final_refined_valleys else (np.array([]),np.array([]))
    
    return (rpx, rpy), (rvx, rvy)


# --- SECTION 4: MAIN ORCHESTRATING FUNCTION (No Segmentation - v3) ---
def process_spectrum_for_lfc_lines_v3(
    y_axis_orig, x_axis_orig=None,
    # ... (all other parameters as before) ...
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    clustering_and_feature_params: dict = None,
    plot_main_details=False, plot_spectrum_and_derivs_flag=True, plot_clustering_details_flag=True,
    refine_to_original_params: dict = None,
    verbose=False
):
    x_axis_orig, y_axis_orig = _datacheck(x_axis_orig, y_axis_orig)
    if verbose: print(f"Processing spectrum of length {len(y_axis_orig)}")

    # --- Default parameters ---
    # ... (default parameter setup as in your last provided code) ...
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust','user_val':None,'gw_params':{'target_min_period_pixels':10,'target_max_period_pixels':20,'default_window_period':15,'overall_min_period_pixels':5,'verbose':verbose}}
    global_noise_val = robust_noise_std(y_axis_orig - np.median(y_axis_orig))
    if clustering_and_feature_params is None:
        clustering_and_feature_params = {'dbscan_eps':0.5,'dbscan_min_samples':5,'min_features_for_clustering':8,'min_extrema_for_spacing_model':4,'global_noise_std_for_fallback':global_noise_val,'noise_estimation_window_pixels_feat':15,'spacing_poly_degree_feat':2,'spacing_uncertainty_const_feat':2.0,'feature_scaling_options':{'x':'minmax','prom_snr':'robust','spacing_dev':'robust'}}
    elif 'global_noise_std_for_fallback' not in clustering_and_feature_params:
        clustering_and_feature_params['global_noise_std_for_fallback'] = global_noise_val
    if refine_to_original_params is None:
        refine_to_original_params = {
            'y_refinement_method': 'flux_weighted_mean', # 'parabolic_vertex', 'parabolic_at_barycenter'
            'y_refinement_weights': [0.25, 0.5, 0.25] # Only used if method is 'weighted_mean'
        }


    # --- Stage 1: Supersample, Smooth ---
    if verbose: print("--- Stage 1: Initial Peak/Valley Detection on Full Spectrum ---")
    if super_sample_factor > 1 and len(y_axis_orig) > 0 :
        y_rebinned = _rebin(y_axis_orig, newshape=(int(super_sample_factor * len(y_axis_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_orig), np.max(x_axis_orig), len(y_rebinned))
    else:
        y_rebinned = np.copy(y_axis_orig); x_rebinned = np.copy(x_axis_orig)

    if len(y_rebinned) == 0: print("Rebinned data is empty."); return ([[], []], [[], []])

    plot_gw = plot_main_details and plot_spectrum_and_derivs_flag
    win_method = window_len_method_options['method']
    gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose
    if win_method == 'auto_robust':
        window_len_on_orig_scale = get_window_robust_targeted(y_axis_orig, plot=plot_gw, **gw_p)
    elif win_method == 'auto_original_get_window': # You'll need to define get_window if you use this
        window_len_on_orig_scale = get_window(y_axis_orig, plot=plot_gw)
    elif win_method == 'user_defined' and window_len_method_options.get('user_val') is not None:
        window_len_on_orig_scale = mathfunc.round_down_to_odd(window_len_method_options['user_val'])
    else:
        window_len_on_orig_scale = gw_p.get('default_window_period',15)
        if verbose: print(f"    Using default window {window_len_on_orig_scale} for full spectrum.")

    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Orig scale window: {window_len_on_orig_scale}, Rebinned smooth window: {actual_smoothing_window_len}")
    
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])

    # --- Call the new function here ---
    initial_peaks_x, initial_peaks_y, \
    initial_valleys_x, initial_valleys_y = detect_initial_extrema_candidates( # <<< MODIFIED CALL
        y_smoothed, x_coords_for_smoothed, deriv_method, first_deriv_zero_threshold_factor
    )
    if verbose: print(f"  Initial derivative candidates: {len(initial_peaks_x)}P, {len(initial_valleys_x)}V.")
    # --- End of modification ---

    if not initial_peaks_x.size and not initial_valleys_x.size:
        print("No initial candidates from derivatives. Exiting.")
        return ([[], []], [[], []])


    # --- Stage 2: Apply Combined Clustering Filter ---
    if verbose: print(f"\n--- Stage 2: Applying Combined Clustering Filter ---")
    (clustered_peaks_x, clustered_peaks_y, clustered_valleys_x, clustered_valleys_y,
     plot_data_combined) = apply_clustering_filter_to_extrema_v3(
        initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
        clustering_and_feature_params['dbscan_eps'],
        clustering_and_feature_params['dbscan_min_samples'],
        clustering_and_feature_params['min_features_for_clustering'],
        y_axis_orig, x_axis_orig,
        clustering_and_feature_params['global_noise_std_for_fallback'],
        clustering_and_feature_params['noise_estimation_window_pixels_feat'],
        clustering_and_feature_params['spacing_poly_degree_feat'],
        clustering_and_feature_params['spacing_uncertainty_const_feat'],
        clustering_and_feature_params.get('feature_scaling_options', None)
    )
    if verbose: print(f"  After clustering: {len(clustered_peaks_x)}P, {len(clustered_valleys_x)}V.")

    final_peaks_x_arr, final_peaks_y_arr = clustered_peaks_x, clustered_peaks_y
    final_valleys_x_arr, final_valleys_y_arr = clustered_valleys_x, clustered_valleys_y

    # --- Stage 3: Refine ALL clustered extrema ---
    if refine_to_original_params.get('search_window_orig_pixels', -1) >= 0:
        if verbose: print(f"\n--- Stage 3: Refining all {len(final_peaks_x_arr)}P / {len(final_valleys_x_arr)}V to original data ---")
        # refine_extrema_to_original_data expects tuples of (x_list, y_list)
        (r_p_x, r_p_y), (r_v_x, r_v_y) = refine_extrema_to_original_data(
            (final_peaks_x_arr, final_peaks_y_arr), # Pass y from smoothed as placeholder
            (final_valleys_x_arr, final_valleys_y_arr),
            x_axis_orig, y_axis_orig,
            # search_window_orig_pixels is not used for x-pos in this version of refine,
            # but y-refinement uses points around the mapped x.
            y_refinement_method=refine_to_original_params.get('y_refinement_method', 'weighted_mean'),
            y_refinement_weights=refine_to_original_params.get('y_refinement_weights', [0.25, 0.5, 0.25])
        )
        final_peaks_x_refined, final_peaks_y_refined = r_p_x, r_p_y
        final_valleys_x_refined, final_valleys_y_refined = r_v_x, r_v_y
        if verbose: print(f"    After y-value refinement: {len(final_peaks_x_refined)}P, {len(final_valleys_x_refined)}V.")
    else: # No refinement, use results from clustering (y-values are from smoothed data)
        if verbose: print("\n--- Stage 3: Skipping y-value refinement to original data ---")
    
    if final_peaks_x_arr.size > 0 or final_valleys_x_arr.size > 0 :
        final_peaks_x_arr, final_peaks_y_arr, \
        final_valleys_x_arr, final_valleys_y_arr = _ensure_alternation(
            (final_peaks_x_arr, final_peaks_y_arr), (final_valleys_x_arr, final_valleys_y_arr)
        )

    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        # ... (plotting logic as before, ensuring derivatives are available for plot_smoothing_and_derivatives) ...
        # Calculate derivatives again for plotting if they were not passed out of detect_initial_extrema_candidates
        # Or modify detect_initial_extrema_candidates to return them.
        # For now, assume they are the `deriv1` and `deriv2` calculated above.
        cand_x_for_plot = plot_data_combined[3]
        cand_y_for_plot = plot_data_combined[4]
        cand_types_for_plot = plot_data_combined[5]
        cand_y_plot_orig_scale = np.interp(cand_x_for_plot, x_axis_orig, y_axis_orig, left=np.nan, right=np.nan) if cand_x_for_plot.size > 0 else np.array([])


        if plot_spectrum_and_derivs_flag:
            deriv1 = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method) # Recalculate for safety or pass
            deriv2 = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1)) if deriv1.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_orig, y_axis_orig, x_coords_for_smoothed, y_smoothed,
                deriv1, deriv2, # Use derivatives calculated earlier in this function
                initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        if plot_clustering_details_flag:
            plot_clustering_2d_results(
                x_axis_orig, y_axis_orig,
                x_coords_for_smoothed, y_smoothed,
                cand_x_for_plot[cand_types_for_plot==1], cand_y_plot_orig_scale[cand_types_for_plot==1], 
                cand_x_for_plot[cand_types_for_plot==-1], cand_y_plot_orig_scale[cand_types_for_plot==-1],
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr
            )
            if plot_data_combined[1] is not None:
                 plot_feature_space_3d_v2(
                     scaled_features=plot_data_combined[1],
                     labels=plot_data_combined[2],
                     original_types=cand_types_for_plot,
                     title_suffix="Full Spectrum"
                 )

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]

def process_spectrum_for_lfc_lines_v4(
    y_axis_orig_input, x_axis_orig_input=None,
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    # New dictionary for rule-based filtering parameters
    rule_filter_params: dict = None,
    y_refinement_on_candidates_params: dict = None,
    plot_main_details=False, plot_spectrum_and_derivs_flag=True, plot_filtering_details_flag=True, # Renamed from clustering
    verbose=False
):
    x_axis_true_orig, y_axis_true_orig = _datacheck(x_axis_orig_input, y_axis_orig_input)
    if verbose: print(f"Processing spectrum (len {len(y_axis_true_orig)}) with Rule-Based Filter")

    # --- Default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust','user_val':None,'gw_params':{'target_min_period_pixels':10,'target_max_period_pixels':20,'default_window_period':15,'overall_min_period_pixels':5,'verbose':verbose}}
    
    if y_refinement_on_candidates_params is None:
        y_refinement_on_candidates_params = {'y_refinement_method':'weighted_mean', 'y_refinement_weights':[0.25,0.5,0.25]}

    # Parameters for the new rule-based filter
    global_noise_val = robust_noise_std(y_axis_true_orig - np.median(y_axis_true_orig))
    if rule_filter_params is None:
        rule_filter_params = {
            'spacing_poly_degree': 2, 'spacing_uncertainty_const': 2.0,
            'spacing_max_dev_factor': 3.0, 'envelope_poly_degree': 3, # Lowered default envelope degree
            'envelope_ratio_poly_degree': 2, 'envelope_ratio_target_min': 0.01,
            'envelope_ratio_target_max': 0.08, 'envelope_ratio_max_dev': 0.03, # Allow more deviation
            'local_snr_min_thresh': 3.0, 'noise_estimation_window_pixels': 15,
            'global_fallback_noise_std': global_noise_val, 'verbose': verbose
        }
    elif 'global_fallback_noise_std' not in rule_filter_params:
        rule_filter_params['global_fallback_noise_std'] = global_noise_val
    if 'verbose' not in rule_filter_params : rule_filter_params['verbose'] = verbose


    # --- Stage 1: Initial Peak/Valley X and Smoothed Y Detection ---
    # ... (Supersampling, Smoothing, detect_initial_extrema_candidates as in process_spectrum_for_lfc_lines_v4) ...
    if verbose: print("--- Stage 1: Initial Peak/Valley X and Smoothed Y ---")
    if super_sample_factor > 1 and len(y_axis_true_orig) > 0 :
        y_rebinned = _rebin(y_axis_true_orig, newshape=(int(super_sample_factor * len(y_axis_true_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_true_orig), np.max(x_axis_true_orig), len(y_rebinned))
    else: y_rebinned = np.copy(y_axis_true_orig); x_rebinned = np.copy(x_axis_true_orig)
    if len(y_rebinned) == 0: print("Rebinned data empty."); return ([[], []], [[], []])

    plot_gw = plot_main_details and plot_spectrum_and_derivs_flag
    win_method = window_len_method_options['method']; gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose # Pass verbose
    if win_method == 'auto_robust': window_len_on_orig_scale = get_window_robust_targeted(y_axis_true_orig, plot=plot_gw, **gw_p)
    else: window_len_on_orig_scale = gw_p.get('default_window_period',15)
    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Smooth window: {actual_smoothing_window_len} (on rebinned)")
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])

    initial_peaks_x, initial_peaks_y_smoothed, \
    initial_valleys_x, initial_valleys_y_smoothed = \
        detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed, deriv_method)
    if verbose: print(f"  Initial derivative cands: {len(initial_peaks_x)}P (y from smooth), {len(initial_valleys_x)}V (y from smooth).")
    if not initial_peaks_x.size and not initial_valleys_x.size: print("No initial cands."); return ([[], []], [[], []])

    # --- Stage 1.5: Refine Y-values of these candidates using ORIGINAL data ---
    if verbose: print("\n--- Stage 1.5: Refining Y-values of initial candidates on original data ---")
    _ , initial_peaks_y_refined_orig = refine_extrema_y_values_on_original(
        initial_peaks_x, x_axis_true_orig, y_axis_true_orig, **y_refinement_on_candidates_params)
    _ , initial_valleys_y_refined_orig = refine_extrema_y_values_on_original(
        initial_valleys_x, x_axis_true_orig, y_axis_true_orig, **y_refinement_on_candidates_params)
    if verbose: print(f"    Y-values refined. {len(initial_peaks_y_refined_orig)}P, {len(initial_valleys_y_refined_orig)}V.")

    # --- Stage 2: Apply Rule-Based Filter ---
    if verbose: print(f"\n--- Stage 2: Applying Rule-Based Filter ---")
    final_peaks_x_arr, final_peaks_y_arr, \
    final_valleys_x_arr, final_valleys_y_arr, \
    plot_data_for_2d_filter = filter_lfc_extrema_v4(
        initial_peaks_x, initial_peaks_y_refined_orig,
        initial_valleys_x, initial_valleys_y_refined_orig,
        x_axis_true_orig, y_axis_true_orig, # Pass original full spectrum for context
        **rule_filter_params # Pass all rule-based filtering parameters
    )
    if verbose: print(f"  After rule-based filtering: {len(final_peaks_x_arr)}P, {len(final_valleys_x_arr)}V.")
    
    # Y-values are already refined and from original data. X-values are from smoothed.
    # No further call to refine_extrema_to_original_data is needed for y-values.

    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        cand_x_plot = plot_data_for_2d_filter[3] # all_cand_x from _ensure_alternation within filter_v4
        cand_y_plot_refined = plot_data_for_2d_filter[4] # all_cand_y_refined
        cand_types_plot = plot_data_for_2d_filter[5]    # all_cand_types

        if plot_spectrum_and_derivs_flag:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_true_orig, y_axis_true_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y_smoothed,
                initial_valleys_x, initial_valleys_y_smoothed,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        if plot_filtering_details_flag: # Changed from plot_clustering_details_flag
            # Plotting the effect of the rule-based filter
            plot_filtering_2d_results_v2( # This is your plot_clustering_2d_results_v2 adapted
                x_axis_true_orig, y_axis_true_orig,
                x_coords_for_smoothed, y_smoothed, # Smoothed context
                cand_x_plot[cand_types_plot==1], cand_y_plot_refined[cand_types_plot==1], 
                cand_x_plot[cand_types_plot==-1], cand_y_plot_refined[cand_types_plot==-1],
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr,
                title_suffix="Full Spectrum (Rule-Based Filter)"
            )
            # 3D feature plot is less relevant here as we didn't use DBSCAN on 3 features in this path
            # unless filter_lfc_extrema_v4 was modified to also output some features for plotting.
            # The plot_data_for_2d_filter currently doesn't contain features for 3D plot.

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]


def process_spectrum_for_lfc_lines_v4_rule_based( # Name kept for consistency from previous
    y_axis_orig_input, x_axis_orig_input=None,
    super_sample_factor=3,
    window_len_method_options = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    hybrid_filter_params = None, # Parameters for the rule-based filter
    y_refinement_on_candidates_params = None,
    plot_main_details=False, # Master switch for all plots from this function
    plot_deriv_stage_details=True, # Plot for initial smoothing/derivatives
    plot_filter_stage_details=True,  # Plot for the rule-based filter diagnostics itself
                                     # AND the final 2D summary plot
    verbose=False
):
    x_axis_true_orig, y_axis_true_orig = _datacheck(x_axis_orig_input, y_axis_orig_input)
    if verbose: print(f"Processing spectrum (len {len(y_axis_true_orig)}) with Rule-Based Filter v_revised_plots")

    # --- Default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust','user_val':None,'gw_params':{'target_min_period_pixels':10,'target_max_period_pixels':20,'default_window_period':15,'overall_min_period_pixels':5,'verbose':verbose}}
    
    global_noise_val = robust_noise_std(y_axis_true_orig - np.median(y_axis_true_orig))
    if hybrid_filter_params is None:
        hybrid_filter_params = dict(
            spacing_poly_degree=2, 
            spacing_uncertainty_const=2.0, 
            spacing_max_dev_factor=3.0,
            envelope_poly_degree=3,
            max_allowed_Venv_Penv_ratio = 0.3,
            peak_snr_min_thresh_poisson=5.0, # S/N threshold specifically for peaks
            # Parameters for S/N calculation context
            noise_estimation_window_pixels=15, 
            global_fallback_noise_std=None,
            # Parameters for DBSCAN (applied to survivors of rule-filter)
            dbscan_eps=0.5, dbscan_min_samples=5,
            min_candidates_for_dbscan=10, # If fewer than this survive rules, skip DBSCAN
            feature_scaling_options = None,
            # Control
            plot_rule_diagnostics=False, # Plot diagnostics for the rule-based part
            verbose=False
            )
    elif 'global_fallback_noise_std' not in rule_filter_params:
        rule_filter_params['global_fallback_noise_std'] = global_noise_val
    if 'verbose' not in hybrid_filter_params : hybrid_filter_params['verbose'] = verbose
    # Ensure the sub-plot flag is passed correctly
    hybrid_filter_params['plot_rule_diagnostics'] = plot_main_details and plot_filter_stage_details
    # print(hybrid_filter_params)

    if y_refinement_on_candidates_params is None:
        y_refinement_on_candidates_params = {'y_refinement_method': 'parabolic_vertex', 'y_refinement_weights':None}


    # --- Stage 1: Initial Peak/Valley X and Smoothed Y Detection ---
    # ... (Identical to previous process_spectrum_for_lfc_lines_v4) ...
    if verbose: print("--- Stage 1: Initial Peak/Valley X and Smoothed Y ---")
    # ... (supersampling, smoothing, actual_smoothing_window_len calculation) ...
    if super_sample_factor > 1 and len(y_axis_true_orig) > 0 :
        y_rebinned = _rebin(y_axis_true_orig, newshape=(int(super_sample_factor * len(y_axis_true_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_true_orig), np.max(x_axis_true_orig), len(y_rebinned))
    else: y_rebinned = np.copy(y_axis_true_orig); x_rebinned = np.copy(x_axis_true_orig)
    if len(y_rebinned) == 0: print("Rebinned data empty."); return ([[], []], [[], []])
    plot_gw = plot_main_details and plot_deriv_stage_details
    win_method = window_len_method_options['method']; gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose
    if win_method == 'auto_robust': 
        window_len_on_orig_scale = get_window_robust_targeted(y_axis_true_orig, plot=plot_gw, **gw_p)
    else: window_len_on_orig_scale = gw_p.get('default_window_period',15)
    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Smooth window: {actual_smoothing_window_len} (on rebinned)")
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])
    initial_peaks_x, initial_peaks_y_smoothed, initial_valleys_x, initial_valleys_y_smoothed = \
        detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed, deriv_method)
    if verbose: print(f"  Initial derivative cands: {len(initial_peaks_x)}P (y from smooth), {len(initial_valleys_x)}V (y from smooth).")
    if not initial_peaks_x.size and not initial_valleys_x.size: print("No initial cands."); return ([[], []], [[], []])


    # --- Stage 1.5: Refine Y-values of these candidates using ORIGINAL data ---
    if verbose: print("\n--- Stage 1.5: Refining Y-values of initial candidates on original data ---")
    (initial_peaks_x_ref, initial_peaks_y_refined_orig), \
    (initial_valleys_x_ref, initial_valleys_y_refined_orig) = \
        refine_extrema_to_original_data(
            (initial_peaks_x, initial_peaks_y_smoothed),
            (initial_valleys_x, initial_valleys_y_smoothed),
            x_axis_true_orig, y_axis_true_orig,
            y_refinement_method=y_refinement_on_candidates_params.get('y_refinement_method', 'parabolic_vertex'),
            y_refinement_weights=y_refinement_on_candidates_params.get('y_refinement_weights', [0.25,0.5,0.25])
        )
    if verbose: print(f"    Y-values refined. {len(initial_peaks_y_refined_orig)}P, {len(initial_valleys_y_refined_orig)}V.")
    # --- Stage 2: Apply Rule-Based Filter ---
    print(hybrid_filter_params)
    if verbose: print(f"\n--- Stage 2: Applying Rule-Based Filter (Revised S/N & Plots) ---")
    final_peaks_x_arr, final_peaks_y_arr, \
    final_valleys_x_arr, final_valleys_y_arr, \
    plot_data_for_filter_summary = filter_lfc_extrema_v6_hybrid( # Call the version with plots
        initial_peaks_x_ref, initial_peaks_y_refined_orig,
        initial_valleys_x_ref, initial_valleys_y_refined_orig,
        x_axis_true_orig, y_axis_true_orig,
        **hybrid_filter_params # Pass all rule-based filtering parameters
    )
    if verbose: print(f"  After rule-based filtering: {len(final_peaks_x_arr)}P, {len(final_valleys_x_arr)}V.")
    
    # Y-values are already refined. X-values are from smoothed data.
    
    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        # plot_data_for_filter_summary = (None,None,None, cand_x, cand_y_refined, cand_types) from filter_lfc_extrema...
        cand_x_plot = plot_data_for_filter_summary[3] 
        cand_y_plot_refined = plot_data_for_filter_summary[4]
        cand_types_plot = plot_data_for_filter_summary[5]

        if plot_deriv_stage_details:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_true_orig, y_axis_true_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y_smoothed, # Initial from smoothed
                initial_valleys_x, initial_valleys_y_smoothed,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        
        # The filter_lfc_extrema_v4_revised_with_plots already produces its own diagnostic plots
        # if rule_filter_params['plot_filter_diagnostics'] is True.
        # We might still want a final summary plot.
        if plot_filter_stage_details: # This flag now also controls the *summary* 2D plot
            print("\nPlotting final summary of filtering:")
            plot_filtering_2d_results_v2( # Generic 2D results plotter
                x_axis_true_orig, y_axis_true_orig,
                x_coords_for_smoothed, y_smoothed, # Smoothed context
                cand_x_plot[cand_types_plot==1], cand_y_plot_refined[cand_types_plot==1], 
                cand_x_plot[cand_types_plot==-1], cand_y_plot_refined[cand_types_plot==-1],
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr,
                title_suffix="Full Spectrum (Final after Rule Filter)"
            )
            # 3D feature plot for DBSCAN is not relevant here, as this path doesn't use that style of feature clustering.
            # If filter_lfc_extrema_v4_revised_with_plots was modified to output values for each criterion,
            # we could plot distributions of those.

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]


# --- Main Orchestrating Function (v6 Hybrid) ---
def process_spectrum_for_lfc_lines_v6_hybrid(
    y_axis_orig_input, x_axis_orig_input=None,
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    # Consolidated parameters for the hybrid filter
    hybrid_filter_params: dict = None,
    y_refinement_on_candidates_params: dict = None,
    plot_main_details=False,
    plot_deriv_stage_details=True,
    plot_filter_stage_details=True, # This controls both rule diagnostics AND final summary plots
    verbose=False
):
    x_axis_true_orig, y_axis_true_orig = _datacheck(x_axis_orig_input, y_axis_orig_input)
    if verbose: print(f"Processing spectrum (len {len(y_axis_true_orig)}) with Hybrid Filter v6")

    # --- Default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust','user_val':None,'gw_params':{'target_min_period_pixels':10,'target_max_period_pixels':20,'default_window_period':15,'overall_min_period_pixels':5,'verbose':verbose}}
    
    global_noise_val = robust_noise_std(y_axis_true_orig - np.median(y_axis_true_orig))
    if hybrid_filter_params is None:
        hybrid_filter_params = {
            # Rule-based params
            'spacing_poly_degree': 2, 
            'spacing_uncertainty_const': 2.0,
            'spacing_max_dev_factor': 3.0, 
            'max_allowed_Venv_Penv_ratio': 0.5, # Simplified ratio
            'peak_snr_min_thresh_poisson': 3.0, # Adjusted threshold
            'noise_estimation_window_pixels': 15,
            'global_fallback_noise_std': global_noise_val,
            # DBSCAN params (for survivors of rules)
            'dbscan_eps': 0.5, 'dbscan_min_samples': 5,
            'min_candidates_for_dbscan': 10,
            'feature_scaling_options': {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'},
            # Control flags for filter_lfc_extrema_v6_hybrid
            'plot_filter_diagnostics': plot_main_details and plot_filter_stage_details,
            'verbose': verbose
        }
    # Ensure these flags are correctly propagated
    hybrid_filter_params.setdefault('plot_filter_diagnostics', plot_main_details and plot_filter_stage_details)
    hybrid_filter_params.setdefault('verbose', verbose)
    if 'global_fallback_noise_std' not in hybrid_filter_params: # Ensure it's there
        hybrid_filter_params['global_fallback_noise_std'] = global_noise_val


    if y_refinement_on_candidates_params is None:
        y_refinement_on_candidates_params = {'y_refinement_method': 'parabolic_vertex'}

    # --- Stage 1: Initial Peak/Valley X and Smoothed Y Detection ---
    # ... (Identical to process_spectrum_for_lfc_lines_v4_rule_based) ...
    if verbose: print("--- Stage 1: Initial Peak/Valley X and Smoothed Y ---")
    if super_sample_factor > 1 and len(y_axis_true_orig) > 0 :
        y_rebinned = _rebin(y_axis_true_orig, newshape=(int(super_sample_factor * len(y_axis_true_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_true_orig), np.max(x_axis_true_orig), len(y_rebinned))
    else: y_rebinned = np.copy(y_axis_true_orig); x_rebinned = np.copy(x_axis_true_orig)
    if len(y_rebinned) == 0: print("Rebinned data empty."); return ([[], []], [[], []])
    plot_gw = plot_main_details and plot_deriv_stage_details
    win_method = window_len_method_options['method']; gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose
    if win_method == 'auto_robust': window_len_on_orig_scale = get_window_robust_targeted(y_axis_true_orig, plot=plot_gw, **gw_p)
    else: window_len_on_orig_scale = gw_p.get('default_window_period',15)
    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Smooth window: {actual_smoothing_window_len} (on rebinned)")
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])
    initial_peaks_x, initial_peaks_y_smoothed, initial_valleys_x, initial_valleys_y_smoothed = \
        detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed, deriv_method)
    if verbose: print(f"  Initial derivative cands: {len(initial_peaks_x)}P (y from smooth), {len(initial_valleys_x)}V (y from smooth).")
    if not initial_peaks_x.size and not initial_valleys_x.size: print("No initial cands."); return ([[], []], [[], []])

    # --- Stage 1.5: Refine Y-values of these candidates using ORIGINAL data ---
    if verbose: print("\n--- Stage 1.5: Refining Y-values of initial candidates on original data ---")
    (initial_peaks_x_ref, initial_peaks_y_refined_orig), \
    (initial_valleys_x_ref, initial_valleys_y_refined_orig) = \
        refine_extrema_to_original_data(
            (initial_peaks_x, initial_peaks_y_smoothed),
            (initial_valleys_x, initial_valleys_y_smoothed),
            x_axis_true_orig, y_axis_true_orig,
            **y_refinement_on_candidates_params
        )
    if verbose: print(f"    Y-values refined. {len(initial_peaks_y_refined_orig)}P, {len(initial_valleys_y_refined_orig)}V.")

    # --- Stage 2: Apply Hybrid Rule-Based + DBSCAN Filter ---
    if verbose: print(f"\n--- Stage 2: Applying Hybrid Filter (Rules then DBSCAN) ---")
    final_peaks_x_arr, final_peaks_y_arr, \
    final_valleys_x_arr, final_valleys_y_arr, \
    plot_data_final_summary = filter_lfc_extrema_v6_hybrid( # Call the new hybrid filter
        initial_peaks_x_ref, initial_peaks_y_refined_orig,
        initial_valleys_x_ref, initial_valleys_y_refined_orig,
        x_axis_true_orig, y_axis_true_orig,
        **hybrid_filter_params # Pass all hybrid filtering parameters
    )
    # Y-values are already refined. X-values are from smoothed data.
    
    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        # plot_data_final_summary = (unscaled_feat_dbscan, scaled_feat_dbscan, labels_dbscan, 
        #                            cand_x_to_dbscan, cand_y_to_dbscan, cand_types_to_dbscan)
        # OR if DBSCAN was skipped, it's based on rule-filter inputs.
        
        cand_x_for_summary_plot = plot_data_final_summary[3] 
        cand_y_for_summary_plot = plot_data_final_summary[4] 
        cand_types_for_summary_plot = plot_data_final_summary[5]

        if plot_deriv_stage_details:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_true_orig, y_axis_true_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y_smoothed, # Initial from smoothed
                initial_valleys_x, initial_valleys_y_smoothed,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        
        # The filter_lfc_extrema_v6_hybrid handles its own rule-diagnostic plots if its flag is set.
        # Now plot the final summary using plot_filtering_2d_results_v2
        # This plot shows the candidates that went into the *final DBSCAN stage* (or into rules if DBSCAN skipped)
        # vs the *absolute final* output.
        if plot_filter_stage_details: # This flag also controls this final summary plot
            print("\nPlotting final summary of filtering (candidates to final DBSCAN vs. final output):")
            plot_filtering_2d_results_v2( 
                x_axis_true_orig, y_axis_true_orig,
                x_coords_for_smoothed, y_smoothed, # Smoothed context
                cand_x_for_summary_plot[cand_types_for_summary_plot==1], cand_y_for_summary_plot[cand_types_for_summary_plot==1], 
                cand_x_for_summary_plot[cand_types_for_summary_plot==-1], cand_y_for_summary_plot[cand_types_for_summary_plot==-1],
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr,
                title_suffix="Full Spectrum (Final after Hybrid Filter)"
            )
            # And the 3D feature plot for the DBSCAN stage, if it ran
            if plot_data_final_summary[1] is not None: # scaled_features_combined from DBSCAN stage
                 plot_feature_space_3d_v2(
                     scaled_features=plot_data_final_summary[1], 
                     labels=plot_data_final_summary[2],
                     original_types=cand_types_for_summary_plot, # Types of points fed to DBSCAN
                     title_suffix="Full Spectrum (DBSCAN Stage)"
                 )

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]


def process_spectrum_for_lfc_lines_v7(
    y_axis_orig_input, x_axis_orig_input=None,
    super_sample_factor=3,
    window_len_method_options: dict = None,
    deriv_method='coeff', first_deriv_zero_threshold_factor=0.03,
    # Consolidated parameters for the hybrid filter (rules + separate DBSCAN)
    hybrid_filter_params: dict = None,
    triplet_removal_params: dict = None,
    y_refinement_on_candidates_params: dict = None,
    plot_main_details=False,
    plot_deriv_stage_details=True, # Plot for initial smoothing/derivatives
    plot_filter_stage_details=True,  # Plot for rule-based filter diagnostics
    plot_dbscan_stage_details=True, # Plot for final DBSCAN stage (2D summary and 3D features)
    verbose=False
):
    x_axis_true_orig, y_axis_true_orig = _datacheck(x_axis_orig_input, y_axis_orig_input)
    if verbose: print(f"Processing spectrum (len {len(y_axis_true_orig)}) with Hybrid Filter v7 (Separate DBSCAN)")

    # --- Default parameters ---
    if window_len_method_options is None:
        window_len_method_options = {'method':'auto_robust',
                                     'user_val':None,
                                     'gw_params':{'target_min_period_pixels':10,
                                                  'target_max_period_pixels':35,
                                                  'default_window_period':15,
                                                  'overall_min_period_pixels':5,
                                                  'verbose':verbose}}
    
    global_noise_val = robust_noise_std(y_axis_true_orig - np.median(y_axis_true_orig))
    if hybrid_filter_params is None:
        hybrid_filter_params = {
            'spacing_poly_degree': 1, 'spacing_uncertainty_const': 2.0,
            'spacing_max_dev_factor': 3.0, 'envelope_savgol_window': 51,
            'envelope_savgol_polyorder': 3, #'max_allowed_Venv_Penv_ratio': 0.35,
            'ratio_trend_savgol_window':101, 'ratio_trend_savgol_polyorder': 2,
            'peak_snr_min_thresh_poisson': 3.0,
            'noise_estimation_window_pixels': 15,
            'global_fallback_noise_std': global_noise_val,
            # 'dbscan_eps_peaks': 0.5, 'dbscan_min_samples_peaks': 4, # DBSCAN for Peaks
            # 'dbscan_eps_valleys': 0.5, 'dbscan_min_samples_valleys': 4, # DBSCAN for Valleys
            # 'min_candidates_for_dbscan': 10,
            # 'feature_scaling_options_peaks': {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'},
            # 'feature_scaling_options_valleys': {'x':'minmax','prom_snr':'robust','spacing_dev':'robust'},
            # 'spacing_poly_degree_feat': 2, # For feature calc within DBSCAN stage
            # 'spacing_uncertainty_const_feat': 2.0, # For feature calc
            'plot_filter_diagnostics': plot_main_details and plot_filter_stage_details, # For rules part
            'verbose': verbose
        }
    # Ensure critical params have defaults if dict is partially provided
    hybrid_filter_params.setdefault('global_fallback_noise_std', global_noise_val)
    hybrid_filter_params.setdefault('plot_filter_diagnostics', plot_main_details and plot_filter_stage_details)
    hybrid_filter_params.setdefault('verbose', verbose)


    if y_refinement_on_candidates_params is None:
        y_refinement_on_candidates_params = {'y_refinement_method': 'weighted_mean'}

    # --- Stage 1: Initial Peak/Valley X (from smooth) and Y (from smooth) Detection ---
    if verbose: print("--- Stage 1: Initial Peak/Valley X and Smoothed Y ---")
    if super_sample_factor > 1 and len(y_axis_true_orig) > 0 :
        y_rebinned = _rebin(y_axis_true_orig, newshape=(int(super_sample_factor * len(y_axis_true_orig)),))
        x_rebinned = np.linspace(np.min(x_axis_true_orig), np.max(x_axis_true_orig), len(y_rebinned))
    else: y_rebinned = np.copy(y_axis_true_orig); x_rebinned = np.copy(x_axis_true_orig)
    if len(y_rebinned) == 0: print("Rebinned data empty."); return ([[], []], [[], []])

    plot_gw = plot_main_details and plot_deriv_stage_details # Control get_window plot
    win_method = window_len_method_options['method']; gw_p = window_len_method_options.get('gw_params', {})
    gw_p['verbose'] = verbose # Pass verbose to get_window
    if win_method == 'auto_robust': window_len_on_orig_scale = get_window_robust_targeted(y_axis_true_orig, plot=plot_gw, **gw_p)
    # ... (other window selection methods as before) ...
    else: window_len_on_orig_scale = gw_p.get('default_window_period',15)
    
    actual_smoothing_window_len = mathfunc.round_down_to_odd(int(window_len_on_orig_scale * super_sample_factor))
    if len(y_rebinned) > 0:
        max_poss_win = mathfunc.round_down_to_odd(len(y_rebinned)//2-1 if len(y_rebinned)//2-1>=1 else 1)
        if max_poss_win < 3 and len(y_rebinned) >=3 : max_poss_win = 3
        actual_smoothing_window_len = min(actual_smoothing_window_len, max_poss_win)
    actual_smoothing_window_len = max(3, actual_smoothing_window_len)
    if verbose: print(f"  Smooth window: {actual_smoothing_window_len} (on rebinned)")
    
    y_smoothed = _smooth(y_rebinned, window_len=actual_smoothing_window_len, window='nuttall', mode="same")
    x_coords_for_smoothed = x_rebinned
    if len(y_smoothed) < 3: print("Smoothed data too short."); return ([[], []], [[], []])

    initial_peaks_x, initial_peaks_y_smoothed, \
    initial_valleys_x, initial_valleys_y_smoothed = \
        detect_initial_extrema_candidates(y_smoothed, x_coords_for_smoothed, deriv_method)
    if verbose: print(f"  Initial derivative cands: {len(initial_peaks_x)}P, {len(initial_valleys_x)}V (Y from smooth).")
    if not initial_peaks_x.size and not initial_valleys_x.size: print("No initial cands."); return ([[], []], [[], []])

    # --- Stage 1.5: Refine Y-values of initial candidates using ORIGINAL data ---
    if verbose: print("\n--- Stage 1.5: Refining Y-values of initial candidates on original data ---")
    initial_peaks_x, initial_peaks_y_refined_orig = refine_extrema_to_original_pixel_grid(
        initial_peaks_x, x_axis_true_orig, y_axis_true_orig, **y_refinement_on_candidates_params)
    initial_valleys_x, initial_valleys_y_refined_orig = refine_extrema_to_original_pixel_grid(
        initial_valleys_x, x_axis_true_orig, y_axis_true_orig, **y_refinement_on_candidates_params)
    if verbose: print(f"    Y-values refined. {len(initial_peaks_y_refined_orig)}P, {len(initial_valleys_y_refined_orig)}V.")
    # Note: initial_peaks_x_ref is same as initial_peaks_x (x-pos not changed by y-refinement func)


    # --- NEW Stage 1.75: Remove False Triplets ---
    # First, combine peaks and valleys from Stage 1.5 using _ensure_alternation_and_tag
    # to get a single list suitable for remove_false_triplets
    # This also ensures initial strict alternation before triplet removal.

    # Parameters for triplet removal (should come from a new param dict or defaults)
    if triplet_removal_params is None:
        triplet_removal_params = {
            'max_triplet_x_span_pixels': window_len_on_orig_scale * 0.5, # e.g., 30% of typical period
            'min_prom_depth_factor_center': 2, # Center must be at least 10% of outer span/height
            'y_consistency_factor':0.2,
        }
    
    # max_triplet_x_span = triplet_removal_params.get('max_triplet_x_span_pixels', window_len_on_orig_scale * 0.3)
    # min_prom_depth_factor_triplet = triplet_removal_params.get('min_prominence_depth_factor_for_center', 0.1)
    if verbose: print("\n--- Stage 1.75: Removing False Triplets ---")
    
    _, _, _, _, \
    combined_x_for_triplet_filt, \
    combined_y_for_triplet_filt, \
    combined_types_for_triplet_filt = _ensure_alternation_and_tag(
        initial_peaks_x, initial_peaks_y_refined_orig, # Using X from smooth, Y from original
        initial_valleys_x, initial_valleys_y_refined_orig
    )

    if combined_x_for_triplet_filt.size > 0:
        # remove_false_triplets returns separated lists AND combined lists
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y, \
        valleys_after_triplet_filt_x, valleys_after_triplet_filt_y, \
        _, _, _ = remove_false_triplets_v2( # We only need the separated lists from its return
            combined_x_for_triplet_filt,
            combined_y_for_triplet_filt,
            combined_types_for_triplet_filt,
            **triplet_removal_params,
            verbose=verbose
        )
        if verbose: print(f"    After triplet removal: {len(peaks_after_triplet_filt_x)}P, {len(valleys_after_triplet_filt_x)}V.")
    else: # No candidates to feed to triplet filter
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y = initial_peaks_x, initial_peaks_y_refined_orig
        valleys_after_triplet_filt_x, valleys_after_triplet_y = initial_valleys_x, initial_valleys_y_refined_orig
        
    # --- Stage 2: Apply Hybrid Rule-Based Pre-Filter + Separate DBSCAN ---
    if verbose: print(f"\n--- Stage 2: Applying Hybrid Filter (Rules then Separate DBSCAN) ---")
    final_peaks_x_arr, final_peaks_y_arr, \
    final_valleys_x_arr, final_valleys_y_arr, \
    plot_data_output_hybrid = filter_lfc_extrema_v8_final_rules_with_plots( 
        peaks_after_triplet_filt_x, peaks_after_triplet_filt_y,
        valleys_after_triplet_filt_x, valleys_after_triplet_filt_y,
        x_axis_true_orig, y_axis_true_orig,
        **hybrid_filter_params # Pass all hybrid filtering parameters
    )
    # Y-values from here (final_peaks_y_arr, etc.) are already refined to original.
    # X-values are from the smoothed data.
    
    print(f"\n--- Overall Processing Complete ---")
    print(f"Found total {len(final_peaks_x_arr)} peaks & {len(final_valleys_x_arr)} valleys.")

    # --- Plotting ---
    if plot_main_details:
        # plot_data_output_hybrid = (peaks_dbscan_plot_info, valleys_dbscan_plot_info)
        # each _plot_info = (unscaled_feat, scaled_feat, labels, cand_x, cand_y, types_array_for_those_cands)
        peaks_plot_info = plot_data_output_hybrid[0]
        valleys_plot_info = plot_data_output_hybrid[1]

        if plot_deriv_stage_details:
            deriv1_plot = mathfunc.derivative1d(y_smoothed, order=1, method=deriv_method)
            deriv2_plot = mathfunc.derivative1d(y_smoothed, order=2, method=deriv_method)
            deriv1_abs_thresh_plot = first_deriv_zero_threshold_factor * np.max(np.abs(deriv1_plot)) if deriv1_plot.size > 0 else 0
            plot_smoothing_and_derivatives(
                x_axis_true_orig, y_axis_true_orig, x_coords_for_smoothed, y_smoothed,
                deriv1_plot, deriv2_plot,
                initial_peaks_x, initial_peaks_y_smoothed,
                initial_valleys_x, initial_valleys_y_smoothed,
                actual_smoothing_window_len, deriv1_abs_thresh_plot
            )
        
        # The filter_lfc_extrema_v7_hybrid_sep_dbscan handles its own rule-diagnostic plots
        # if hybrid_filter_params['plot_filter_diagnostics'] is True.

        # Plot final summary (candidates to DBSCAN vs. final DBSCAN output)
        if plot_filter_stage_details: # Reuse this flag for the final summary after DBSCAN
            print("\nPlotting final summary of filtering (candidates to final DBSCAN stage vs. final output):")
            
            # Candidates that went into DBSCAN for peaks (i.e., rule-survivors for peaks)
            cand_p_x_dbscan = peaks_plot_info[3]
            cand_p_y_dbscan = peaks_plot_info[4] # These Ys are already refined
            # Candidates that went into DBSCAN for valleys
            cand_v_x_dbscan = valleys_plot_info[3]
            cand_v_y_dbscan = valleys_plot_info[4] # These Ys are already refined
            
            plot_filtering_2d_results_v2( # Your generic 2D results plotter
                x_axis_true_orig, y_axis_true_orig,
                x_coords_for_smoothed, y_smoothed, # Smoothed context
                cand_p_x_dbscan, cand_p_y_dbscan, 
                cand_v_x_dbscan, cand_v_y_dbscan,
                final_peaks_x_arr, final_peaks_y_arr, 
                final_valleys_x_arr, final_valleys_y_arr,
                title_suffix="Full Spectrum (Final after Hybrid Filter)"
            )

        # Plot 3D feature spaces for DBSCAN stages if data exists
        if plot_dbscan_stage_details:
            if peaks_plot_info[1] is not None: # scaled_features for peaks
                 plot_feature_space_3d_v2(
                     scaled_features=peaks_plot_info[1], 
                     labels=peaks_plot_info[2],
                     original_types=peaks_plot_info[5], # Should be all +1s
                     title_suffix="Peak DBSCAN Stage"
                 )
            if valleys_plot_info[1] is not None: # scaled_features for valleys
                 plot_feature_space_3d_v2(
                     scaled_features=valleys_plot_info[1], 
                     labels=valleys_plot_info[2],
                     original_types=valleys_plot_info[5], # Should be all -1s
                     title_suffix="Valley DBSCAN Stage"
                 )

    return [final_peaks_x_arr.tolist(), final_peaks_y_arr.tolist()], \
           [final_valleys_x_arr.tolist(), final_valleys_y_arr.tolist()]


# --- Plotting Utilities (Updated for consistency) ---
def plot_clustering_2d_results(
    x_axis_orig, y_axis_orig, # For main background plot
    x_coords_smoothed, y_values_smoothed, # For plotting smoothed data as context
    cand_peaks_x, cand_peaks_y_orig_scale,   # Candidates (y-values now on original scale)
    cand_valleys_x, cand_valleys_y_orig_scale,
    final_peaks_x, final_peaks_y, 
    final_valleys_x, final_valleys_y,
    title_suffix=""
):
    plt.figure(figsize=(15, 7))
    main_title = "Clustering Filter Results"
    if title_suffix: main_title += f" - {title_suffix}"
    plt.title(main_title, fontsize=14)

    plt.plot(x_axis_orig, y_axis_orig, label='Original Data', color='lightgrey', alpha=0.7, zorder=1)
    if x_coords_smoothed is not None and y_values_smoothed is not None:
        plt.plot(x_coords_smoothed, y_values_smoothed, label='Smoothed Data (Context)', color='gray', alpha=0.4, linestyle=':', zorder=1)

    # Ensure inputs are arrays for .size
    cand_peaks_x, cand_peaks_y = np.asarray(cand_peaks_x), np.asarray(cand_peaks_y_orig_scale)
    cand_valleys_x, cand_valleys_y = np.asarray(cand_valleys_x), np.asarray(cand_valleys_y_orig_scale)
    final_peaks_x, final_peaks_y = np.asarray(final_peaks_x), np.asarray(final_peaks_y)
    final_valleys_x, final_valleys_y = np.asarray(final_valleys_x), np.asarray(final_valleys_y)

    # Plot candidates (y-values are now on original scale)
    if cand_peaks_x.size > 0:
        plt.scatter(cand_peaks_x, cand_peaks_y, color='pink', marker='o', s=50, label='Peak Cand. (to Clust.)', zorder=2, alpha=0.6)
    if cand_valleys_x.size > 0:
        plt.scatter(cand_valleys_x, cand_valleys_y, color='lightblue', marker='o', s=50, label='Valley Cand. (to Clust.)', zorder=2, alpha=0.6)

    # For identifying discarded, compare X-COORDINATES ONLY from candidates to final
    set_final_peaks_x = set(np.round(final_peaks_x, decimals=5))
    set_final_valleys_x = set(np.round(final_valleys_x, decimals=5))

    false_clust_peaks_x, false_clust_peaks_y = [], []
    if cand_peaks_x.size > 0:
        for x, y in zip(cand_peaks_x, cand_peaks_y): # cand_peaks_y is already on orig scale
            if round(x, 5) not in set_final_peaks_x:
                false_clust_peaks_x.append(x)
                false_clust_peaks_y.append(y)
    
    false_clust_valleys_x, false_clust_valleys_y = [], []
    if cand_valleys_x.size > 0:
        for x, y in zip(cand_valleys_x, cand_valleys_y): # cand_valleys_y is already on orig scale
            if round(x, 5) not in set_final_valleys_x:
                false_clust_valleys_x.append(x)
                false_clust_valleys_y.append(y)
    
    plt.scatter(false_clust_peaks_x, false_clust_peaks_y, color='red', marker='x', s=100, label='Discarded Peaks (Clust.)', zorder=3)
    plt.scatter(false_clust_valleys_x, false_clust_valleys_y, color='magenta', marker='x', s=100, label='Discarded Valleys (Clust.)', zorder=3)

    if final_peaks_x.size > 0: plt.scatter(final_peaks_x, final_peaks_y, edgecolor='blue', facecolor='none', marker='o', s=120, label='Final Peaks', zorder=4, linewidth=1.5)
    if final_valleys_x.size > 0: plt.scatter(final_valleys_x, final_valleys_y, edgecolor='green', facecolor='none', marker='o', s=120, label='Final Valleys', zorder=4, linewidth=1.5)
    
    plt.xlabel('X-coordinate (Original Scale)'); plt.ylabel('Flux (Original Scale)'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.7); plt.show()

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
    
def plot_filtering_2d_results_v2( # Keep the content same, just rename for context
    x_axis_orig, y_axis_orig,
    x_coords_smoothed, y_values_smoothed,
    cand_peaks_x, cand_peaks_y_refined_orig,
    cand_valleys_x, cand_valleys_y_refined_orig,
    final_peaks_x, final_peaks_y_refined,
    final_valleys_x, final_valleys_y_refined,
    title_suffix=""
):
    # ... (Identical implementation to plot_clustering_2d_results_v2 from previous response) ...
    # ... It correctly plots candidates vs final, using x-coordinates for "discarded" logic ...
    plt.figure(figsize=(15, 7))
    main_title = "Filtering Results" # Generic title
    if title_suffix: main_title += f" - {title_suffix}"
    plt.title(main_title, fontsize=14)
    plt.plot(x_axis_orig,y_axis_orig,label='Original Data',color='lightgrey',alpha=0.7,zorder=1)
    if x_coords_smoothed is not None and y_values_smoothed is not None and x_coords_smoothed.size>0:
        plt.plot(x_coords_smoothed,y_values_smoothed,label='Smoothed Data (Context)',color='gray',alpha=0.4,linestyle=':',zorder=1)
    cpk_x,cpk_y=np.asarray(cand_peaks_x),np.asarray(cand_peaks_y_refined_orig)
    cvl_x,cvl_y=np.asarray(cand_valleys_x),np.asarray(cand_valleys_y_refined_orig)
    fpk_x,fpk_y=np.asarray(final_peaks_x),np.asarray(final_peaks_y_refined)
    fvl_x,fvl_y=np.asarray(final_valleys_x),np.asarray(final_valleys_y_refined)
    if cpk_x.size>0:plt.scatter(cpk_x,cpk_y,color='pink',marker='o',s=50,label='Peak Cand. (to Filter)',zorder=2,alpha=0.6)
    if cvl_x.size>0:plt.scatter(cvl_x,cvl_y,color='lightblue',marker='o',s=50,label='Valley Cand. (to Filter)',zorder=2,alpha=0.6)
    sfpk_x=set(np.round(fpk_x,decimals=5));sfvl_x=set(np.round(fvl_x,decimals=5))
    false_pk_x,false_pk_y=[],[]
    if cpk_x.size>0:
        for x,y in zip(cpk_x,cpk_y):
            if round(x,5) not in sfpk_x:false_pk_x.append(x);false_pk_y.append(y)
    false_vl_x,false_vl_y=[],[]
    if cvl_x.size>0:
        for x,y in zip(cvl_x,cvl_y):
            if round(x,5) not in sfvl_x:false_vl_x.append(x);false_vl_y.append(y)
    plt.scatter(false_pk_x,false_pk_y,color='red',marker='x',s=100,label='Discarded Peaks (Filter)',zorder=3)
    plt.scatter(false_vl_x,false_vl_y,color='magenta',marker='x',s=100,label='Discarded Valleys (Filter)',zorder=3)
    if fpk_x.size>0:plt.scatter(fpk_x,fpk_y,edgecolor='blue',facecolor='none',marker='o',s=120,label='Final Peaks',zorder=4,linewidth=1.5)
    if fvl_x.size>0:plt.scatter(fvl_x,fvl_y,edgecolor='green',facecolor='none',marker='o',s=120,label='Final Valleys',zorder=4,linewidth=1.5)
    plt.xlabel('X (Orig Scale)');plt.ylabel('Flux (Orig Scale)');plt.legend();plt.grid(True,ls=':',alpha=0.7);plt.show()