#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:19:51 2025

@author: dmilakov
"""

import numpy as np
import fitsio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import traceback
import warnings
import time

# Use relative imports within the package
from .io import load_echelle_data
from .peaks import find_peaks
from .clustering import cluster_peaks_to_orders
from .fitting import twoD_Gaussian
from .plotting import (plot_gaussian_ellipse, plot_raw_data_stamps,
                       plot_normalized_residuals_stamps) # Removed comparison plots - called from run script

try:
    from .zernike_fitter import ZernikeFitter, ZERNPY_AVAILABLE, generate_zernike_indices
except ImportError:
    warnings.warn("zernike_fitter module not found. Zernike fitting will be skipped.")
    ZERNPY_AVAILABLE = False
    # Define dummy ZernikeFitter if import fails
    class ZernikeFitter:
         def __init__(self, *args, **kwargs): self.n_coeffs=0; self.zernike_indices = []
         def fit(self, *args, **kwargs): self.success=False; self.message="Fitter unavailable"; return False
         def get_results(self, *args, **kwargs): return {'success': False}
         def plot_power_spectrum(self, *args, **kwargs): pass
         def plot_fit_comparison(self, *args, **kwargs): pass
         def plot_fit_residuals(self, *args, **kwargs): pass
    def generate_zernike_indices(n_max): return []


class EchelleAnalyzer:
    """
    Analyzes echelle spectrograph data to fit Zernike polynomials to stacked PSF profiles.
    Handles data loading, peak finding, order clustering, segmented analysis,
    and saving results to a FITS file using variable-length arrays.
    """
    def __init__(self, lfc_filename, bias_filename=None, output_dir='./echelle_analysis_output',
                 # Peak Finding Params
                 peak_min_distance=5, peak_threshold_abs=None, peak_threshold_rel=None, peak_hard_cut=None, # Added hard_cut
                 # Clustering Params
                 cluster_eps_x=0.002, cluster_eps_y=0.02, cluster_min_samples=5, # Adjusted defaults based on user code
                 # Stamp/Fit Params
                 stamp_half_width=5, fit_threshold_snr=5.0,
                 # Segmentation Params
                 num_segments=16,
                 # Zernike Params
                 n_max_zern=6, r_max_zern=5.0,
                 # Output Control
                 output_suffix='_zernike_results.fits', clobber_output=True):
        """ Initializes the EchelleAnalyzer. """
        self.lfc_path = Path(lfc_filename)
        self.bias_path = Path(bias_filename) if bias_filename else None
        self.output_dir = Path(output_dir)
        self.output_fits_path = None
        self.output_suffix = output_suffix
        self.clobber_output = clobber_output

        # Store parameters (using user's clustering defaults)
        self.params = {
            'peak_min_distance': peak_min_distance, 'peak_threshold_abs': peak_threshold_abs,
            'peak_threshold_rel': peak_threshold_rel, 'peak_hard_cut': peak_hard_cut, # Store hard_cut
            'cluster_eps_x': cluster_eps_x, 'cluster_eps_y': cluster_eps_y,
            'cluster_min_samples': cluster_min_samples, 'stamp_half_width': stamp_half_width,
            'fit_threshold_snr': fit_threshold_snr, 'num_segments': num_segments,
            'n_max_zern': n_max_zern, 'r_max_zern': r_max_zern,
        }

        self.image_data = None; self.image_shape = None; self.detector = None
        self.hdu_index = None; self.all_peaks_xy = None; self.paired_orders_dict = None
        self.zernike_indices = None
        self.results_table_dtype = None # Will be defined later

        # Attempt to define dtype right away if possible
        self._define_results_dtype()

    def _define_results_dtype(self):
        """Defines the numpy dtype for the results table."""
        num_coeffs = 0
        # Need generate_zernike_indices (should be imported from zernike_fitter)
        try:
             self.zernike_indices = generate_zernike_indices(self.params['n_max_zern'])
             num_coeffs = len(self.zernike_indices)
        except NameError:
             self.zernike_indices = []
             warnings.warn("generate_zernike_indices not found. Cannot determine coefficient dimensions.")

        base_dtype = [
            ('ORDER_NUM', 'i4'), ('IMGTYPE', 'i2'), ('SEGMENT', 'i4'),
            ('N_MAX_ZERN', 'i2'), ('R_MAX_ZERN', 'f4'),
            ('NUM_PEAKS_PROC', 'i4'), ('NUM_PIX_STACKED', 'i4'),
            ('MEDIAN_X', 'f4'), ('MEDIAN_Y', 'f4'), 
            ('FIT_SUCCESS', 'bool'),
            ('RMSE', 'f4'), ('R_SQUARED', 'f4')
        ]
        coeff_dtype = []
        if num_coeffs > 0:
             coeff_dtype = [('COEFFS', f'{num_coeffs}f4'), ('ERR_COEFFS', f'{num_coeffs}f4')]

        # Use object dtype for var-len arrays - fitsio handles this well
        varlen_dtype = [('X_STACK', 'O'), ('Y_STACK', 'O'), ('Z_STACK', 'O')]
        self.results_table_dtype = np.dtype(base_dtype + coeff_dtype + varlen_dtype)


    def load_data(self, detector='red'):
        """Loads data for the specified detector ('red' or 'blue')."""
        self.detector = detector.lower()
        # Use user's HDU logic
        self.hdu_index = 2 if self.detector == 'red' else 1
        print(f"\nLoading data for detector: {self.detector} (HDU {self.hdu_index})...")
        try:
            self.image_data = load_echelle_data(self.lfc_path, self.bias_path, hdu_index=self.hdu_index)
            self.image_shape = self.image_data.shape
            print(f"Data loaded successfully. Shape: {self.image_shape}")
            self.output_fits_path = self.output_dir / f"{self.lfc_path.stem}_{self.detector}{self.output_suffix}"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Output FITS will be saved to: {self.output_fits_path}")
            return True
        except FileNotFoundError as e: print(f"Error: {e}"); self.image_data = None; return False
        except Exception as e: print(f"Error loading data: {e}"); traceback.print_exc(); self.image_data = None; return False

    def find_all_peaks(self, detector='red', cut=None):
        """Finds peaks in the loaded image data."""
        if self.image_data is None: print("Error: Image data not loaded."); return False
        print("\nFinding peaks...")
        hard_flux_cut = cut if cut is not None else 320 if detector == 'red' else 420
        
        try:
            self.all_peaks_xy = find_peaks(
                self.image_data,
                min_distance=self.params['peak_min_distance'],
                hard_cut= hard_flux_cut,
                plot_interactive=False
            )
            if self.all_peaks_xy is None or len(self.all_peaks_xy) == 0: print("Warning: No peaks found."); return False
            print(f"Found {len(self.all_peaks_xy)} peaks.")
            return True
        except Exception as e: print(f"Error during peak finding: {e}"); traceback.print_exc(); self.all_peaks_xy = None; return False

    def cluster_orders(self):
        """Clusters peaks into echelle orders and identifies A/B pairs."""
        if self.all_peaks_xy is None: print("Error: Peaks not found."); return False
        if self.image_shape is None: print("Error: Image shape not known."); return False
        print("\nClustering peaks into orders...")
        try:
            _, self.paired_orders_dict = cluster_peaks_to_orders(
                self.all_peaks_xy, self.image_shape,
                eps_x=self.params['cluster_eps_x'], eps_y=self.params['cluster_eps_y'],
                min_samples=self.params['cluster_min_samples'],
                plot_interactive=True,
            )
            if not self.paired_orders_dict: print("Warning: No orders/clusters found."); return False
            print(f"Found {len(self.paired_orders_dict)} potential order pairs.")

            # --- Calculate and store median positions ---
            print("Calculating median positions for segments...")
            self.median_segment_positions = {}
            num_seg = self.params['num_segments']
            for order_num, pair_info in self.paired_orders_dict.items():
                 for img_type_str, peaks_full_order in pair_info.items():
                     if peaks_full_order is None or len(peaks_full_order) == 0: continue
                     img_type_int = 0 if img_type_str == 'A' else 1
                     # Sort by X (important for consistent segmentation)
                     peaks_sorted = peaks_full_order[np.argsort(peaks_full_order[:, 0])]
                     segmented_peaks = np.array_split(peaks_sorted, num_seg)
                     for seg_idx, seg_peaks in enumerate(segmented_peaks):
                          if len(seg_peaks) > 0:
                               med_x = np.median(seg_peaks[:, 0])
                               med_y = np.median(seg_peaks[:, 1])
                               self.median_segment_positions[(order_num, img_type_int, seg_idx)] = (med_x, med_y)
                          else: # Store NaN if segment is empty
                               self.median_segment_positions[(order_num, img_type_int, seg_idx)] = (np.nan, np.nan)

            print("Median segment positions calculated.")
            return True
        except Exception as e: print(f"Error during clustering/median calc: {e}"); traceback.print_exc(); self.paired_orders_dict = None; return False

    def _process_single_stamp(self, peak_xy):
        """ Internal helper: Processes a single peak. Returns dict or None. """
        peak_x, peak_y = peak_xy
        sw = self.params['stamp_half_width']
        snr_thresh = self.params['fit_threshold_snr']
        img_h, img_w = self.image_shape

        x_min, x_max = max(0, peak_x - sw), min(img_w, peak_x + sw + 1)
        y_min, y_max = max(0, peak_y - sw), min(img_h, peak_y + sw + 1)
        if (x_max - x_min) != (2*sw + 1) or (y_max - y_min) != (2*sw + 1): return None

        stamp_data = self.image_data[y_min:y_max, x_min:x_max]
        total_flux = np.sum(stamp_data)
        if total_flux <= 1e-9: return None
        stamp_norm = stamp_data / total_flux

        stamp_h, stamp_w = stamp_norm.shape
        y_sg, x_sg = np.meshgrid(np.arange(stamp_h), np.arange(stamp_w), indexing='ij')
        p0 = (np.max(stamp_norm), stamp_w/2.0, stamp_h/2.0, sw/3.0, sw/3.0, 0.0, np.min(stamp_norm))
        bounds = ([0, -stamp_w, -stamp_h, 0.1, 0.1, -np.pi, -np.inf],
                  [np.inf, 2*stamp_w, 2*stamp_h, stamp_w, stamp_h, np.pi, np.inf])
        try:
            popt, pcov = curve_fit(twoD_Gaussian, (x_sg, y_sg), stamp_norm.ravel(), p0=p0, bounds=bounds, maxfev=5000)
            if not np.all(np.isfinite(pcov)): raise RuntimeError("Non-finite cov")
            perr = np.sqrt(np.diag(pcov))
            amp_snr = popt[0] / perr[0] if perr[0] > 1e-12 else np.inf
            if amp_snr < snr_thresh or not np.all(np.isfinite(popt)) or popt[3] <= 0 or popt[4] <= 0: return None
        except (RuntimeError, ValueError): return None

        amp_n, xc_s, yc_s, sx, sy, th, off_n = popt
        xc_abs, yc_abs = x_min + xc_s, y_min + yc_s
        y_ag, x_ag = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
        rel_x, rel_y = x_ag - xc_abs, y_ag - yc_abs
        th_major = th if sx >= sy else th + np.pi / 2.0
        rot_ang = -th_major
        cos_r, sin_r = np.cos(rot_ang), np.sin(rot_ang)
        rel_x_rot = rel_x * cos_r + rel_y * sin_r
        rel_y_rot = -rel_x * sin_r + rel_y * cos_r

        model_norm = twoD_Gaussian((x_sg, y_sg), *popt)
        res_norm_1d = stamp_norm.ravel() - model_norm
        var_norm = np.maximum(stamp_data.ravel(), 1e-9) / (total_flux**2)
        res_div_sig = res_norm_1d / np.sqrt(var_norm)
        chi2 = np.sum(res_div_sig**2)
        dof = max(1, len(stamp_norm.ravel()) - len(popt))
        red_chi2 = chi2 / dof

        return {
            'x_rotated': rel_x_rot.ravel().astype('f4'), # Ensure type
            'y_rotated': rel_y_rot.ravel().astype('f4'),
            'z_norm': stamp_norm.ravel().astype('f4'),
            'plot_info': {
                'stamp_data': stamp_data, 'residuals_norm_2d': res_div_sig.reshape(stamp_h, stamp_w),
                'xo_stamp': xc_s, 'yo_stamp': yc_s, 'sigma_x': sx, 'sigma_y': sy,
                'theta_rad': th, 'chi2_reduced': red_chi2
            }
        }


    def analyze_segment(self, segment_peaks_xy, order_num, img_type_int, segment_idx, plot_config=None):
        """ Analyzes a single segment. Returns results dict or None. """
        if plot_config is None: plot_config = {}
        save_plots = plot_config.get('save_plots', False)
        base_plot_dir = Path(plot_config.get('plot_dir', self.output_dir / "plots")) # Get base plot dir

        # --- Determine Plot Flags ---
        stamps_config = plot_config.get('stamps', {})
        zernike_config = plot_config.get('zernike', {})
        spectrum_config = plot_config.get('spectrum', {}) # Get spectrum config too

        plot_stamps = stamps_config.get('enable', False)
        plot_zernike = zernike_config.get('enable', False)
        plot_spectrum = spectrum_config.get('enable', False) # Check spectrum flag

        print(f"--- Analyzing Order {order_num}, ImgType {img_type_int}, Segment {segment_idx} ({len(segment_peaks_xy)} peaks) ---")
        segment_results_x, segment_results_y, segment_results_z = [], [], []
        plotting_info_list = []
        num_peaks_processed = 0

        # --- Peak Processing Loop ---
        for i, peak_xy in enumerate(segment_peaks_xy):
            processed_stamp = self._process_single_stamp(peak_xy)
            if processed_stamp:
                segment_results_x.append(processed_stamp['x_rotated'])
                segment_results_y.append(processed_stamp['y_rotated'])
                segment_results_z.append(processed_stamp['z_norm'])
                num_peaks_processed += 1
                if plot_stamps: # Only collect if plotting stamps
                     processed_stamp['plot_info']['peak_index'] = i
                     plotting_info_list.append(processed_stamp['plot_info'])
                     
        # --- Get Median Position ---
        # Retrieve pre-calculated median position for this segment
        median_pos = self.median_segment_positions.get((order_num, img_type_int, segment_idx), (np.nan, np.nan))
        median_x, median_y = median_pos

        # --- Handle No Processed Peaks ---
        if not segment_results_x:
            print(f"Segment {segment_idx}: No peaks processed successfully.")
            # Return minimal failure dict matching dtype structure
            coeffs_nan = np.full(len(self.zernike_indices) if self.zernike_indices else 0, np.nan, dtype='f4')
            return {
                'ORDER_NUM': order_num, 'IMGTYPE': img_type_int, 'SEGMENT': segment_idx,
                'N_MAX_ZERN': self.params['n_max_zern'], 'R_MAX_ZERN': self.params['r_max_zern'],
                'NUM_PEAKS_PROC': 0, 'NUM_PIX_STACKED': 0, 
                'MEDIAN_X': median_x, 'MEDIAN_Y': median_y, 
                'FIT_SUCCESS': False, 'RMSE': np.nan, 'R_SQUARED': np.nan,
                'COEFFS': coeffs_nan, 'ERR_COEFFS': coeffs_nan,
                'X_STACK': np.array([], dtype='f4'), 'Y_STACK': np.array([], dtype='f4'), 'Z_STACK': np.array([], dtype='f4')
            }

        # --- Concatenate Data ---
        X_stack = np.concatenate(segment_results_x)
        Y_stack = np.concatenate(segment_results_y)
        Z_stack = np.concatenate(segment_results_z)
        num_pixels_stacked = len(X_stack)
        print(f"Segment {segment_idx}: Processed {num_peaks_processed} peaks, {num_pixels_stacked} stacked pixels.")

        # --- Plotting Setup ---
        img_type_str = 'A' if img_type_int == 0 else 'B' # Convert int back to string
        stamps_final_dir = None
        zernike_final_dir = None

        # --- Construct Plot Directories and Filenames ---
        # Stamps Plots
        if plot_stamps and plotting_info_list:
             custom_subdir = stamps_config.get('subdir')
             if custom_subdir: # If subdir is specified (and not None/empty)
                  stamps_final_dir = base_plot_dir / custom_subdir
             else: # Use default structure
                  stamps_final_dir = base_plot_dir / f"Order{order_num}{img_type_str}" / "stamps"
             stamps_final_dir.mkdir(parents=True, exist_ok=True)

             stamps_filename = stamps_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_stamps.pdf" if save_plots else None
             residuals_filename = stamps_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_norm_residuals.pdf" if save_plots else None
             print(f"Segment {segment_idx}: Generating stamp plots (Save: {save_plots}, Dir: {stamps_final_dir})...")
             try: plot_raw_data_stamps(plotting_info_list, segment_idx, filename=stamps_filename)
             except Exception as e: print(f"ERROR plotting raw stamps: {e}")
             try: plot_normalized_residuals_stamps(plotting_info_list, segment_idx, filename=residuals_filename)
             except Exception as e: print(f"ERROR plotting residual stamps: {e}")

        # --- Zernike Fitting ---
        # (Initialize result variables as before)
        fit_success = False
        coeffs_nan = np.full(len(self.zernike_indices) if self.zernike_indices else 0, np.nan, dtype='f4')
        segment_output = {
            'ORDER_NUM': order_num, 'IMGTYPE': img_type_int, 'SEGMENT': segment_idx,
            'N_MAX_ZERN': self.params['n_max_zern'], 'R_MAX_ZERN': self.params['r_max_zern'],
            'NUM_PEAKS_PROC': num_peaks_processed, 'NUM_PIX_STACKED': num_pixels_stacked,
            'MEDIAN_X': median_x, 'MEDIAN_Y': median_y, 
            'FIT_SUCCESS': False, 'RMSE': np.nan, 'R_SQUARED': np.nan,
            'COEFFS': coeffs_nan, 'ERR_COEFFS': coeffs_nan,
            'X_STACK': X_stack, 'Y_STACK': Y_stack, 'Z_STACK': Z_stack
        }

        if ZERNPY_AVAILABLE and num_pixels_stacked > len(segment_output['COEFFS']) + 2:
            fitter = ZernikeFitter(n_max=self.params['n_max_zern'], r_max=self.params['r_max_zern'])
            # ... (handle potential index mismatch) ...

            ig = {'xc': 0.0, 'yc': 0.0}
            bnds = {'lower': [-0.5, -0.5] + [-np.inf]*fitter.n_coeffs, 'upper': [0.5, 0.5] + [np.inf]*fitter.n_coeffs}
            fit_success = fitter.fit(X_stack, Y_stack, Z_stack, initial_guess=ig, bounds=bnds, verbose=False)
            segment_output['FIT_SUCCESS'] = fit_success

            if fit_success:
                results = fitter.get_results(include_coeffs_table=False)
                segment_output.update({ # Update dict with fit results
                    'RMSE': results.get('rmse', np.nan),
                    'R_SQUARED': results.get('r_squared', np.nan),
                    'COEFFS': results.get('fitted_coeffs', coeffs_nan),
                    'ERR_COEFFS': results.get('err_coeffs', coeffs_nan)
                })
                # ... (Clean NaN arrays as before) ...
                if segment_output['COEFFS'] is None or not isinstance(segment_output['COEFFS'], np.ndarray): segment_output['COEFFS'] = coeffs_nan
                if segment_output['ERR_COEFFS'] is None or not isinstance(segment_output['ERR_COEFFS'], np.ndarray): segment_output['ERR_COEFFS'] = coeffs_nan
                np.nan_to_num(segment_output['COEFFS'], copy=False, nan=np.nan)
                np.nan_to_num(segment_output['ERR_COEFFS'], copy=False, nan=np.nan)

                print(f"Segment {segment_idx}: Zernike fit successful. RMSE={segment_output['RMSE']:.4f}, R^2={segment_output['R_SQUARED']:.4f}")

                # --- Plotting Zernike Results ---
                if plot_zernike or plot_spectrum:
                    # Construct Zernike plot directory
                    custom_subdir = zernike_config.get('subdir') # Check Zernike specific subdir
                    if custom_subdir:
                         zernike_final_dir = base_plot_dir / custom_subdir
                    else:
                         zernike_final_dir = base_plot_dir / f"Order{order_num}{img_type_str}" / "zernike"
                    zernike_final_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Segment {segment_idx}: Generating Zernike plots (Save: {save_plots}, Dir: {zernike_final_dir})...")

                    if plot_zernike:
                          zcomp_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_fit.pdf" if save_plots else None
                          zres_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_residuals.pdf" if save_plots else None
                          title_fit = f"Zernike Fit (O={order_num}{img_type_str}, S={segment_idx})"
                          title_res = f"Zernike Residuals (O={order_num}{img_type_str}, S={segment_idx})"
                          try: fitter.plot_fit_comparison(X_stack, Y_stack, Z_stack, title=title_fit, filename=zcomp_fn)
                          except Exception as e: print(f"ERROR plotting Zernike fit: {e}")
                          try: fitter.plot_fit_residuals(X_stack, Y_stack, Z_stack, title=title_res, filename=zres_fn)
                          except Exception as e: print(f"ERROR plotting Zernike residuals: {e}")

                    if plot_spectrum:
                           spec_type = spectrum_config.get('spectrum_type', 'abs') # Get spectrum type
                           zspec_fn = zernike_final_dir / f"O{order_num}{img_type_str}_S{segment_idx}_zernike_spectrum_{spec_type}.pdf" if save_plots else None # Include type in name
                           title_spec = f"Zernike Spectrum ({spec_type}) (O={order_num}{img_type_str}, S={segment_idx})"
                           try: fitter.plot_power_spectrum(title=title_spec, filename=zspec_fn, plot_type=spec_type)
                           except Exception as e: print(f"ERROR plotting Zernike spectrum: {e}")
            else: print(f"Segment {segment_idx}: Zernike fit failed. {fitter.message}")
        # else: (Handled by print statements outside)

        return segment_output # Return the populated dictionary


    def analyze_order_image(self, order_num, image_type, plot_config=None):
        """Analyzes all segments for a specific order and image type."""
        if self.paired_orders_dict is None: print("Error: Orders not clustered."); return None
        if order_num not in self.paired_orders_dict: print(f"Error: Order {order_num} not found."); return None
        img_type_str = image_type.upper(); img_type_int = 0 if img_type_str == 'A' else 1
        if img_type_str not in ['A', 'B']: print("Error: image_type must be 'A' or 'B'."); return None

        peaks_full_order = self.paired_orders_dict[order_num].get(img_type_str)
        if peaks_full_order is None or len(peaks_full_order) == 0: print(f"Warning: No peaks for O={order_num}, Img={img_type_str}."); return []

        print(f"\n===== Analyzing Order {order_num}, Image {img_type_str} =====")
        peaks_full_order = peaks_full_order[np.argsort(peaks_full_order[:, 0])] # Sort by X
        segmented_peaks = np.array_split(peaks_full_order, self.params['num_segments'])
        results_for_order = []

        for segment_idx, segment_peaks in enumerate(segmented_peaks):
            if len(segment_peaks) < self.params['cluster_min_samples']:
                print(f"--- Skipping Segment {segment_idx} (Too few peaks: {len(segment_peaks)}) ---")
                continue
            segment_result = self.analyze_segment(segment_peaks, order_num, img_type_int, segment_idx, plot_config)
            if segment_result: results_for_order.append(segment_result)

        print(f"===== Finished Order {order_num}, Image {img_type_str} =====")
        return results_for_order

    def analyze_all(self, orders_to_process=None, image_types_to_process=['A', 'B'], plot_config=None):
        """Analyzes all specified orders and image types."""
        if self.paired_orders_dict is None: print("Error: Orders not clustered."); return None
        if orders_to_process is None: orders_to_process = sorted(list(self.paired_orders_dict.keys()))

        all_results = []
        start_time = time.time()
        print(f"\n<<<<< Starting Full Analysis for Orders: {orders_to_process} >>>>>")

        for order_num in orders_to_process:
             if order_num not in self.paired_orders_dict: print(f"Skipping order {order_num} - not found."); continue
             for image_type in image_types_to_process:
                 results = self.analyze_order_image(order_num, image_type, plot_config)
                 if results: all_results.extend(results)

        end_time = time.time()
        print(f"\n<<<<< Full Analysis Complete ({len(all_results)} segments processed) Duration: {end_time - start_time:.2f} sec >>>>>")
        return all_results

    def save_results_to_fits(self, results_list):
        """Saves the collected analysis results to the output FITS file."""
        if not results_list: print("Warning: No results provided to save."); return
        if self.output_fits_path is None: print("Error: Output FITS path not set."); return
        if self.results_table_dtype is None: print("Error: Results table dtype not defined."); return

        print(f"\nSaving results to: {self.output_fits_path}...")
        num_rows = len(results_list)
        data_struct = np.zeros(num_rows, dtype=self.results_table_dtype)
        valid_rows = 0

        # Populate structured array carefully
        for i, row_dict in enumerate(results_list):
             try:
                 for name in self.results_table_dtype.names:
                     if name in row_dict:
                         value = row_dict[name]
                         # Handle var-len arrays (need to be assigned as objects)
                         if name in ['X_STACK', 'Y_STACK', 'Z_STACK']:
                             data_struct[i][name] = np.asarray(value, dtype='f4') # fitsio expects arrays
                         # Handle fixed vector arrays
                         elif name in ['COEFFS', 'ERR_COEFFS']:
                             expected_len = self.results_table_dtype[name].shape[0]
                             current_val = np.asarray(value, dtype='f4')
                             if current_val.shape == (expected_len,):
                                 data_struct[i][name] = current_val
                             else: # Assign NaN array if shape mismatch or invalid
                                 data_struct[i][name] = np.full(expected_len, np.nan, dtype='f4')
                         # Handle scalar values
                         else:
                             data_struct[i][name] = value
                 valid_rows += 1
             except Exception as e:
                  print(f"Error populating row {i} for saving: {e} - Skipping row.")
                  # Optionally fill skipped row with NaNs/defaults if needed later

        if valid_rows == 0: print("Error: No valid rows could be prepared for saving."); return

        # Prepare Headers
        primary_hdr = fitsio.FITSHDR(); results_hdr = fitsio.FITSHDR()
        primary_hdr['ORIGFILE'] = self.lfc_path.name
        primary_hdr['BIASFILE'] = self.bias_path.name if self.bias_path else 'None'
        primary_hdr['DETECTOR'] = self.detector if self.detector else 'Unknown'
        primary_hdr['DATE'] = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
        primary_hdr['AUTHOR'] = 'harps.twodim.analyzer'
        for key, val in self.params.items(): # Add params to header
             hdr_key = key.upper()[:8]; val_str = str(val)
             if len(val_str) < 68: primary_hdr[hdr_key] = val if isinstance(val,(int,float,bool,str)) else val_str
        if self.zernike_indices: # Add Zernike indices to results header
             indices_str = ",".join([f"{n}_{m}" for n, m in self.zernike_indices])
             # Handle long strings if necessary
             max_len = 68; parts = [indices_str[i:i+max_len] for i in range(0, len(indices_str), max_len)]
             for idx, part in enumerate(parts): results_hdr[f'ZNIND{idx}'] = part
        results_hdr['EXTNAME'] = 'SEGMENT_RESULTS'

        # Write FITS
        try:
            with fitsio.FITS(self.output_fits_path, 'rw', clobber=self.clobber_output) as fits:
                fits.write(None, header=primary_hdr)
                # Write the results table - fitsio handles var-len ('O' dtype) correctly
                fits.write(data_struct[:valid_rows], header=results_hdr, extname='SEGMENT_RESULTS')
            print(f"Successfully saved {valid_rows} segment results.")
        except Exception as e: print(f"Error writing FITS file: {e}"); traceback.print_exc()

    # --- Reading Methods --- (Keep as before, they rely on the saved structure)
    def read_results_table(self, fits_path=None):
        # ... (Implementation from previous step) ...
        path = Path(fits_path) if fits_path else self.output_fits_path
        if not path or not path.is_file(): print(f"Error: Results FITS file not found at {path}"); return None
        try:
            with fitsio.FITS(path, 'r') as fits:
                 try:
                      data = fits['SEGMENT_RESULTS'].read()
                      print(f"Read SEGMENT_RESULTS table with {len(data)} rows from {path}.")
                      return data
                 except:
                      try:
                          data = fits[1].read()
                          print(f"Read SEGMENT_RESULTS table with {len(data)} rows from {path}.")
                      except: print(f"Error: HDU 'SEGMENT_RESULTS' not found in {path}"); return None
        except Exception as e: print(f"Error reading FITS file {path}: {e}"); return None


    def get_segment_data(self, order_num, img_type_int, segment_idx, fits_path=None):
        # ... (Implementation from previous step) ...
        results_table = self.read_results_table(fits_path=fits_path)
        if results_table is None: return None, None, None
        mask = (results_table['ORDER_NUM'] == order_num) & (results_table['IMGTYPE'] == img_type_int) & (results_table['SEGMENT'] == segment_idx)
        match_indices = np.where(mask)[0]
        if len(match_indices) == 0: print(f"Data not found for O={order_num},T={img_type_int},S={segment_idx}"); return None, None, None
        segment_row = results_table[match_indices[0]]
        # fitsio reads var-len back as arrays
        return segment_row['X_STACK'], segment_row['Y_STACK'], segment_row['Z_STACK']

    def get_segment_coeffs(self, order_num, img_type_int, segment_idx, fits_path=None):
        # ... (Implementation from previous step) ...
        results_table = self.read_results_table(fits_path=fits_path)
        if results_table is None: return None, None
        mask = (results_table['ORDER_NUM'] == order_num) & (results_table['IMGTYPE'] == img_type_int) & (results_table['SEGMENT'] == segment_idx)
        match_indices = np.where(mask)[0]
        if len(match_indices) == 0: print(f"Coeffs not found for O={order_num},T={img_type_int},S={segment_idx}"); return None, None
        segment_row = results_table[match_indices[0]]
        if 'COEFFS' not in results_table.dtype.names: return None, None
        coeffs = segment_row['COEFFS']
        errs = segment_row.get('ERR_COEFFS') # Use .get for safety if column might be missing
        if coeffs is None or np.all(np.isnan(coeffs)): return None, None # Check for NaN array
        return coeffs, errs