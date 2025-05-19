#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:04:45 2025

@author: dmilakov
"""

import numpy as np
from sklearn.linear_model import TheilSenRegressor # Robust regressor for spacing trend
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from collections import Counter
import matplotlib.pyplot as plt

def _ensure_alternation(peaks_x, peaks_y, valleys_x, valleys_y):
    """
    Ensures strict P-V-P-V alternation.
    If P-P, keeps higher. If V-V, keeps lower.
    Returns numpy arrays.
    """
    # Convert inputs to numpy arrays if they are lists
    peaks_x, peaks_y = np.array(peaks_x), np.array(peaks_y)
    valleys_x, valleys_y = np.array(valleys_x), np.array(valleys_y)

    if not len(peaks_x) and not len(valleys_x): # Both empty
        return np.array([]), np.array([]), np.array([]), np.array([])
    if not len(peaks_x): # Only valleys
        return np.array([]), np.array([]), valleys_x, valleys_y
    if not len(valleys_x): # Only peaks
        return peaks_x, peaks_y, np.array([]), np.array([])

    # Combine all features with a type indicator (1 for peak, -1 for valley)
    all_x = np.concatenate((peaks_x, valleys_x))
    all_y = np.concatenate((peaks_y, valleys_y))
    all_types = np.concatenate((np.ones(len(peaks_x)), -np.ones(len(valleys_x))))

    # Sort by x-coordinate
    sorted_indices = np.argsort(all_x)
    sorted_x = all_x[sorted_indices]
    sorted_y = all_y[sorted_indices]
    sorted_types = all_types[sorted_indices]

    final_x_list, final_y_list, final_types_list = [], [], []

    if len(sorted_x) == 0: # Should not happen if checks above are passed
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Add the first feature
    final_x_list.append(sorted_x[0])
    final_y_list.append(sorted_y[0])
    final_types_list.append(sorted_types[0])

    for i in range(1, len(sorted_x)):
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


def _calculate_features_for_extrema(
    extrema_x, extrema_y,           # Current features (e.g., peaks_x, peaks_y)
    opposite_x, opposite_y,       # Bracketing features (e.g., valleys_x, valleys_y)
    is_peak,                        # Boolean: True if extrema are peaks
    min_extrema_for_spacing_model=3 # Min number of *extrema_x* to attempt spacing model
):
    """
    Calculates features: x_coord, prominence/depth, spacing_deviation.
    Returns a NumPy array of shape (num_extrema, 3).
    """
    num_extrema = len(extrema_x)
    if num_extrema == 0:
        return np.array([]).reshape(0, 3) # Ensure 3 columns for vstack later

    # 1. Prominence / Depth (delta_y)
    prom_depth_values = np.zeros(num_extrema)
    for i in range(num_extrema):
        x_i, y_i = extrema_x[i], extrema_y[i]
        
        # Find bracketing opposite features
        left_indices = np.where(opposite_x < x_i)[0]
        right_indices = np.where(opposite_x > x_i)[0]

        y_brackets_list = []
        if len(left_indices) > 0: # Has a bracketing feature to the left
            y_brackets_list.append(opposite_y[left_indices[-1]]) # Closest on the left
        if len(right_indices) > 0: # Has a bracketing feature to the right
            y_brackets_list.append(opposite_y[right_indices[0]]) # Closest on the right

        if not y_brackets_list: # No bracketing features found
            prom_depth_values[i] = 0 # Assign 0 prominence/depth
            continue

        y_brackets_arr = np.array(y_brackets_list)
        if is_peak:
            # Prominence: y_peak - max(neighboring_valley_y)
            prom_depth_values[i] = y_i - np.max(y_brackets_arr)
        else:
            # Depth: min(neighboring_peak_y) - y_valley
            prom_depth_values[i] = np.min(y_brackets_arr) - y_i
        
        prom_depth_values[i] = max(0, prom_depth_values[i]) # Ensure non-negative


    # 2. Spacing Deviation (delta_x_dev)
    spacing_dev_values = np.zeros(num_extrema)
    # Need at least min_extrema_for_spacing_model points to define enough spacings for a robust fit.
    # TheilSenRegressor typically needs at least 2 points for X and y for a fit.
    # If num_extrema = 3, spacings = 2, spacing_x_midpoints = 2. This is min for TheilSen.
    if num_extrema >= min_extrema_for_spacing_model:
        spacings = np.diff(extrema_x)
        spacing_x_midpoints = (extrema_x[:-1] + extrema_x[1:]) / 2

        # Ensure enough points for the regressor model
        if len(spacing_x_midpoints) >= 2: # At least 2 midpoints (i.e., 3 extrema)
            model = TheilSenRegressor(random_state=42) # Robust to outliers in spacing
            try:
                model.fit(spacing_x_midpoints.reshape(-1, 1), spacings)
                
                for i in range(num_extrema):
                    devs_list = []
                    # Deviation from spacing with previous point
                    if i > 0:
                        d_prev = extrema_x[i] - extrema_x[i-1]
                        x_mid_prev = (extrema_x[i] + extrema_x[i-1]) / 2
                        d_prev_exp = model.predict(np.array([[x_mid_prev]]))[0]
                        devs_list.append(abs(d_prev - d_prev_exp))
                    
                    # Deviation from spacing with next point
                    if i < num_extrema - 1:
                        d_next = extrema_x[i+1] - extrema_x[i]
                        x_mid_next = (extrema_x[i+1] + extrema_x[i]) / 2
                        d_next_exp = model.predict(np.array([[x_mid_next]]))[0]
                        devs_list.append(abs(d_next - d_next_exp))
                    
                    if devs_list:
                        spacing_dev_values[i] = np.mean(devs_list)
                    # else (single point or failed model), spacing_dev_values[i] remains 0
            except ValueError: 
                # Fit can fail if, e.g., x_midpoints are not diverse enough.
                # In this case, spacing_dev_values remain 0.
                pass 
    
    # Assemble the feature matrix: [x_coordinate, prominence/depth, spacing_deviation]
    feature_matrix = np.vstack([
        extrema_x, 
        prom_depth_values,
        spacing_dev_values
    ]).T
    
    return feature_matrix


def filter_extrema_with_clustering(
    initial_peaks_x, initial_peaks_y,
    initial_valleys_x, initial_valleys_y,
    dbscan_eps=0.75,             # DBSCAN epsilon parameter (tune this carefully)
    dbscan_min_samples=5,        # DBSCAN min_samples parameter (tune this)
    n_iterations=1,                # Number of refinement iterations
    min_extrema_for_clustering=10, # Min number of extrema to attempt clustering
    min_extrema_for_spacing_model=3 # Min extrema needed for spacing model fit
):
    """
    Filters peaks and valleys using DBSCAN clustering on a feature space.
    Features: x-coordinate, prominence/depth, spacing deviation.
    Returns two tuples: (filtered_peaks_x, filtered_peaks_y), (filtered_valleys_x, filtered_valleys_y)
    """
    current_peaks_x = np.array(initial_peaks_x)
    current_peaks_y = np.array(initial_peaks_y)
    current_valleys_x = np.array(initial_valleys_x)
    current_valleys_y = np.array(initial_valleys_y)

    # Initial alternation pass
    current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
        _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)

    for iteration_num in range(n_iterations):
        # --- Filter Peaks ---
        num_current_peaks = len(current_peaks_x)
        if num_current_peaks >= min_extrema_for_clustering:
            peak_features = _calculate_features_for_extrema(
                current_peaks_x, current_peaks_y,
                current_valleys_x, current_valleys_y, # Use current valleys for bracketing
                is_peak=True,
                min_extrema_for_spacing_model=min_extrema_for_spacing_model
            )
            if peak_features.shape[0] > 0 and peak_features.shape[1] == 3:
                scaler_peaks = MinMaxScaler()
                scaled_peak_features = scaler_peaks.fit_transform(peak_features)
                
                # Adjust DBSCAN parameters as needed, possibly separately for peaks/valleys
                dbscan_p = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                peak_labels = dbscan_p.fit_predict(scaled_peak_features)

                # Identify the main cluster (largest non -1 cluster)
                core_point_labels = peak_labels[peak_labels != -1] # Labels of points not marked as outliers
                if len(core_point_labels) > 0:
                    # Find the most frequent label among core points
                    most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                    # Select indices of peaks belonging to this main cluster
                    true_peak_indices = np.where(peak_labels == most_common_label)[0]
                    
                    current_peaks_x = current_peaks_x[true_peak_indices]
                    current_peaks_y = current_peaks_y[true_peak_indices]
                elif len(peak_labels) > 0 : # All points are outliers or no clusters formed
                    current_peaks_x, current_peaks_y = np.array([]), np.array([]) # Remove all
            # If not enough features or calculation failed, keep current peaks for this step
        
        # Ensure alternation after peak filtering, before valley filtering
        current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
            _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)

        # --- Filter Valleys ---
        num_current_valleys = len(current_valleys_x)
        if num_current_valleys >= min_extrema_for_clustering:
            valley_features = _calculate_features_for_extrema(
                current_valleys_x, current_valleys_y,
                current_peaks_x, current_peaks_y, # Use (potentially filtered) peaks for bracketing
                is_peak=False,
                min_extrema_for_spacing_model=min_extrema_for_spacing_model
            )
            if valley_features.shape[0] > 0 and valley_features.shape[1] == 3:
                scaler_valleys = MinMaxScaler()
                scaled_valley_features = scaler_valleys.fit_transform(valley_features)

                dbscan_v = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
                valley_labels = dbscan_v.fit_predict(scaled_valley_features)
                
                core_point_labels = valley_labels[valley_labels != -1]
                if len(core_point_labels) > 0:
                    most_common_label = Counter(core_point_labels).most_common(1)[0][0]
                    true_valley_indices = np.where(valley_labels == most_common_label)[0]
                    
                    current_valleys_x = current_valleys_x[true_valley_indices]
                    current_valleys_y = current_valleys_y[true_valley_indices]
                elif len(valley_labels) > 0: # All points are outliers
                    current_valleys_x, current_valleys_y = np.array([]), np.array([])
        
        # Final alternation for this iteration
        current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y = \
            _ensure_alternation(current_peaks_x, current_peaks_y, current_valleys_x, current_valleys_y)
            
    return (list(current_peaks_x), list(current_peaks_y)), \
           (list(current_valleys_x), list(current_valleys_y))
           
           
# --- Visualization Function ---
def visualize_filtering_results(
    x_spectrum_data, y_spectrum_data,
    initial_peaks_x_tuple, initial_peaks_y_tuple,    # These are the raw initial detections
    initial_valleys_x_tuple, initial_valleys_y_tuple, # Before any processing by your filter
    final_peaks_x_list, final_peaks_y_list,          # Output of your filter
    final_valleys_x_list, final_valleys_y_list,
    min_extrema_for_spacing_model_vis=3 # For feature calculation during visualization
):
    """
    Visualizes the spectrum, initial detections, false detections, and final detections.
    Also shows a 3D feature space plot.

    Args:
        x_spectrum_data (list/np.array): X-coordinates of the spectrum.
        y_spectrum_data (list/np.array): Flux values of the spectrum.
        initial_peaks_x_tuple (tuple of lists): Original x-coords of detected peaks.
        initial_peaks_y_tuple (tuple of lists): Original y-coords of detected peaks.
        initial_valleys_x_tuple (tuple of lists): Original x-coords of detected valleys.
        initial_valleys_y_tuple (tuple of lists): Original y-coords of detected valleys.
        final_peaks_x_list (list): Filtered x-coords of peaks.
        final_peaks_y_list (list): Filtered y-coords of peaks.
        final_valleys_x_list (list): Filtered x-coords of valleys.
        final_valleys_y_list (list): Filtered y-coords of valleys.
        min_extrema_for_spacing_model_vis (int): Param for _calculate_features_for_extrema.
    """
    initial_peaks_x = np.array(initial_peaks_x_tuple[0] if isinstance(initial_peaks_x_tuple, tuple) else initial_peaks_x_tuple)
    initial_peaks_y = np.array(initial_peaks_y_tuple[0] if isinstance(initial_peaks_y_tuple, tuple) else initial_peaks_y_tuple)
    initial_valleys_x = np.array(initial_valleys_x_tuple[0] if isinstance(initial_valleys_x_tuple, tuple) else initial_valleys_x_tuple)
    initial_valleys_y = np.array(initial_valleys_y_tuple[0] if isinstance(initial_valleys_y_tuple, tuple) else initial_valleys_y_tuple)
    
    final_peaks_x = np.array(final_peaks_x_list)
    final_peaks_y = np.array(final_peaks_y_list)
    final_valleys_x = np.array(final_valleys_x_list)
    final_valleys_y = np.array(final_valleys_y_list)

    # --- 1. Identify False Detections ---
    # Use a tolerance for floating point comparison if necessary
    # For simplicity, we'll convert to sets of tuples (x,y)
    # Rounding to avoid float precision issues when checking membership
    set_final_peaks = set(zip(np.round(final_peaks_x, decimals=5), np.round(final_peaks_y, decimals=5)))
    set_final_valleys = set(zip(np.round(final_valleys_x, decimals=5), np.round(final_valleys_y, decimals=5)))

    false_peaks_x, false_peaks_y = [], []
    true_initial_peaks_x, true_initial_peaks_y = [], [] # Initial peaks that were kept
    for x, y in zip(initial_peaks_x, initial_peaks_y):
        if (round(x, 5), round(y, 5)) not in set_final_peaks:
            false_peaks_x.append(x)
            false_peaks_y.append(y)
        else:
            true_initial_peaks_x.append(x)
            true_initial_peaks_y.append(y)
    
    false_valleys_x, false_valleys_y = [], []
    true_initial_valleys_x, true_initial_valleys_y = [], [] # Initial valleys that were kept
    for x, y in zip(initial_valleys_x, initial_valleys_y):
        if (round(x, 5), round(y, 5)) not in set_final_valleys:
            false_valleys_x.append(x)
            false_valleys_y.append(y)
        else:
            true_initial_valleys_x.append(x)
            true_initial_valleys_y.append(y)

    # --- 2. 2D Plot: Spectrum with Detections ---
    plt.figure(figsize=(15, 7))
    plt.plot(x_spectrum_data, y_spectrum_data, label='Spectrum Data', color='gray', alpha=0.7, zorder=1)
    
    # Initial Detections (those that were kept, but shown as initial)
    plt.scatter(true_initial_peaks_x, true_initial_peaks_y, color='cyan', marker='o', s=50, label='Initial Peaks (Kept)', zorder=2, alpha=0.6)
    plt.scatter(true_initial_valleys_x, true_initial_valleys_y, color='lime', marker='o', s=50, label='Initial Valleys (Kept)', zorder=2, alpha=0.6)

    # False Detections
    plt.scatter(false_peaks_x, false_peaks_y, color='red', marker='x', s=100, label='False Peaks', zorder=3)
    plt.scatter(false_valleys_x, false_valleys_y, color='magenta', marker='x', s=100, label='False Valleys', zorder=3)

    # Final Detections (can be plotted over the "kept initial" ones for emphasis)
    plt.scatter(final_peaks_x, final_peaks_y, edgecolor='blue', facecolor='none', marker='o', s=120, label='Final Peaks', zorder=4, linewidth=1.5)
    plt.scatter(final_valleys_x, final_valleys_y, edgecolor='green', facecolor='none', marker='o', s=120, label='Final Valleys', zorder=4, linewidth=1.5)
    
    plt.title('Spectrum with Initial, False, and Final Detections')
    plt.xlabel('X-coordinate')
    plt.ylabel('Flux')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

    # --- 3. 3D Feature Space Plot ---
    # We need to calculate features for the initial set of peaks/valleys
    # Apply alternation first to mimic the start of the filtering process
    alt_initial_peaks_x, alt_initial_peaks_y, \
    alt_initial_valleys_x, alt_initial_valleys_y = _ensure_alternation(
        initial_peaks_x, initial_peaks_y, initial_valleys_x, initial_valleys_y
    )

    if len(alt_initial_peaks_x) > 0 and len(alt_initial_valleys_x) > 0: # Need both for prominence calc
        # Features for initial (alternated) peaks
        features_initial_peaks = _calculate_features_for_extrema(
            alt_initial_peaks_x, alt_initial_peaks_y,
            alt_initial_valleys_x, alt_initial_valleys_y, # Use alternated initial valleys as opposites
            is_peak=True,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model_vis
        )
        if features_initial_peaks.shape[0] > 0:
            scaler_peaks = MinMaxScaler()
            scaled_features_peaks = scaler_peaks.fit_transform(features_initial_peaks)

            # Determine which of these alt_initial_peaks survived
            peak_status = [] # True if kept, False if discarded
            set_final_peaks_coords = set(zip(np.round(final_peaks_x, decimals=5))) # Only x for matching features
            
            # This matching is a bit tricky because _ensure_alternation might merge/remove points.
            # We'll match based on the x-coordinate of the alternated initial peaks.
            current_alt_peaks_set = set(zip(np.round(alt_initial_peaks_x, 5), np.round(alt_initial_peaks_y, 5)))

            for x_alt, y_alt in zip(alt_initial_peaks_x, alt_initial_peaks_y):
                 if (round(x_alt, 5), round(y_alt, 5)) in set_final_peaks:
                     peak_status.append('Kept (Final)') # Was in final_peaks
                 else:
                     peak_status.append('Discarded')
            
            colors_peaks = ['blue' if s == 'Kept (Final)' else 'red' for s in peak_status]

            fig_peaks = plt.figure(figsize=(10, 8))
            ax_peaks = fig_peaks.add_subplot(111, projection='3d')
            # scaled_features_peaks columns: 0:x_coord, 1:prominence, 2:spacing_dev
            scatter_peaks = ax_peaks.scatter(
                scaled_features_peaks[:, 0], # Scaled X
                scaled_features_peaks[:, 1], # Scaled Prominence
                scaled_features_peaks[:, 2], # Scaled Spacing Deviation
                c=colors_peaks,
                marker='o'
            )
            ax_peaks.set_xlabel('Scaled X-coordinate')
            ax_peaks.set_ylabel('Scaled Prominence')
            ax_peaks.set_zlabel('Scaled Spacing Deviation')
            ax_peaks.set_title('3D Feature Space for Initial Peaks (after alternation)')
            # Create dummy scatter artists for legend
            kept_proxy = plt.Line2D([0],[0], linestyle="none", c='blue', marker='o')
            discarded_proxy = plt.Line2D([0],[0], linestyle="none", c='red', marker='o')
            ax_peaks.legend([kept_proxy, discarded_proxy], ['Kept', 'Discarded'], numpoints=1)
            plt.show()

    if len(alt_initial_valleys_x) > 0 and len(alt_initial_peaks_x) > 0: # Need both for depth calc
        # Features for initial (alternated) valleys
        features_initial_valleys = _calculate_features_for_extrema(
            alt_initial_valleys_x, alt_initial_valleys_y,
            alt_initial_peaks_x, alt_initial_peaks_y, # Use alternated initial peaks as opposites
            is_peak=False,
            min_extrema_for_spacing_model=min_extrema_for_spacing_model_vis
        )
        if features_initial_valleys.shape[0] > 0:
            scaler_valleys = MinMaxScaler()
            scaled_features_valleys = scaler_valleys.fit_transform(features_initial_valleys)

            valley_status = []
            current_alt_valleys_set = set(zip(np.round(alt_initial_valleys_x, 5), np.round(alt_initial_valleys_y, 5)))

            for x_alt, y_alt in zip(alt_initial_valleys_x, alt_initial_valleys_y):
                if (round(x_alt, 5), round(y_alt, 5)) in set_final_valleys:
                    valley_status.append('Kept (Final)')
                else:
                    valley_status.append('Discarded')

            colors_valleys = ['green' if s == 'Kept (Final)' else 'magenta' for s in valley_status]

            fig_valleys = plt.figure(figsize=(10, 8))
            ax_valleys = fig_valleys.add_subplot(111, projection='3d')
            scatter_valleys = ax_valleys.scatter(
                scaled_features_valleys[:, 0], # Scaled X
                scaled_features_valleys[:, 1], # Scaled Depth
                scaled_features_valleys[:, 2], # Scaled Spacing Deviation
                c=colors_valleys,
                marker='o'
            )
            ax_valleys.set_xlabel('Scaled X-coordinate')
            ax_valleys.set_ylabel('Scaled Depth')
            ax_valleys.set_zlabel('Scaled Spacing Deviation')
            ax_valleys.set_title('3D Feature Space for Initial Valleys (after alternation)')
            kept_proxy = plt.Line2D([0],[0], linestyle="none", c='green', marker='o')
            discarded_proxy = plt.Line2D([0],[0], linestyle="none", c='magenta', marker='o')
            ax_valleys.legend([kept_proxy, discarded_proxy], ['Kept', 'Discarded'], numpoints=1)
            plt.show()
