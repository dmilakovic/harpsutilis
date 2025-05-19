#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 14:42:08 2025

@author: dmilakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to perform a detailed Bayesian fit on a single LFC stamp,
using results from a previous EchelleAnalyzer run saved in a FITS file.
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings
import json 
import re   

# Import necessary components from your package
from harps.twodim.analyzer import EchelleAnalyzer # Needed for reading methods & getting stamp data
from harps.twodim.stampfitter import BayesianStampFitter # The new Bayesian fitter
def parse_zernike_string(zern_str):
    """ Parses string like '(n1,m1) (n2,m2)...' into list of tuples. """
    if not zern_str:
        return None
    indices = []
    # Regex to find pairs of numbers within parentheses, allowing whitespace
    pattern = re.compile(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)')
    matches = pattern.findall(zern_str)
    if not matches:
        raise ValueError(f"Could not parse Zernike string: '{zern_str}'. Expected format like '(n,m) (n,m)'.")
    for n_str, m_str in matches:
        try:
            indices.append((int(n_str), int(m_str)))
        except ValueError:
            raise ValueError(f"Invalid number found in Zernike pair: '({n_str},{m_str})'")
    return indices

def main():
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Fit a single LFC stamp using Bayesian modeling, configured via JSON.")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file.")
    # Add overrides for convenience if needed, e.g., target selection
    parser.add_argument("-ord", "--order", type=int, help="Override target echelle order number from JSON.")
    parser.add_argument("-img", "--image", type=str, choices=['A', 'B'], help="Override target image type ('A' or 'B') from JSON.")
    parser.add_argument("-seg", "--segment", type=int, help="Override target segment index (0-based) from JSON.")
    parser.add_argument("-p", "--peak_index", type=int, help="Override index of the peak *within the segment* from JSON.")
    parser.add_argument("--show_plots", action='store_true', help="Show plots interactively, overriding 'save_plots' in JSON.")
    # Maybe add MCMC overrides?
    # parser.add_argument("--samples", type=int, help="Override MCMC samples.")

    args = parser.parse_args()

    # --- Load Configuration from JSON ---
    config_path = Path(args.config_file)
    if not config_path.is_file():
        print(f"Error: Configuration file not found at {config_path}")
        return
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error reading JSON config file: {e}")
        return

    # --- Get Parameters from Config (using .get for safety) ---
    # File Paths
    results_fits_path = Path(config.get('results_fits_file', ''))
    lfc_file_path = Path(config.get('lfc_source_file', ''))
    bias_file_path = Path(config['bias_file']) if config.get('bias_file') else None
    plot_dir = Path(config.get('plot_settings', {}).get('plot_dir', './single_stamp_plots'))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Target Selection (Allow overrides from command line)
    target_order = args.order if args.order is not None else config.get('target', {}).get('order')
    target_image_str = args.image if args.image is not None else config.get('target', {}).get('image', 'A').upper()
    target_segment = args.segment if args.segment is not None else config.get('target', {}).get('segment')
    target_peak_index = args.peak_index if args.peak_index is not None else config.get('target', {}).get('peak_index')

    if None in [target_order, target_segment, target_peak_index]:
         print("Error: Target order, segment, and peak_index must be specified in JSON or via command line.")
         return
    if target_image_str not in ['A', 'B']:
         print("Error: Target image type must be 'A' or 'B'.")
         return
    img_type_int = 0 if target_image_str == 'A' else 1

    # Fitter Configuration
    fitter_config = config.get('fitter_params', {})
    nmax_stamp = fitter_config.get('nmax_stamp', 4)
    npoly_cont = fitter_config.get('npoly_cont', 6)
    zerns_str = fitter_config.get('zerns_to_fit', None) # Get string from JSON
    try: # Parse the Zernike string
        zernike_indices_to_fit = parse_zernike_string(zerns_str)
        print(f"Fitting Zernikes: {zernike_indices_to_fit if zernike_indices_to_fit else 'Default (all except piston)'}")
    except ValueError as e:
         print(f"Error parsing 'zerns_to_fit' from config: {e}. Using default.")
         zernike_indices_to_fit = None

    # MCMC Configuration
    mcmc_config = config.get('mcmc_params', {})
    num_warmup = mcmc_config.get('warmup', 500)
    num_samples = mcmc_config.get('samples', 1000)
    seed = mcmc_config.get('seed', 0)
    init_strategy = mcmc_config.get('init_strategy', 'median')

    # Plotting Configuration
    plot_settings = config.get('plot_settings', {})
    save_plots = plot_settings.get('save_plots', True)
    if args.show_plots: save_plots = False # Override if show_plots flag is set
    plot_format = plot_settings.get('plot_format', 'png').lower()
    if plot_format not in ['pdf', 'png']: plot_format = 'png'
    enable_plots = not plot_settings.get('no_plots', False) # Check global disable flag

    # --- Initialize Analyzer (for reading) ---
    print(f"\nInitializing reader for results file: {results_fits_path}")
    # Pass dummy LFC path if not strictly needed by methods used (get_fits_metadata might need it)
    reader_analyzer = EchelleAnalyzer(lfc_filename=lfc_file_path or 'dummy.fits', bias_filename=bias_file_path)
    reader_analyzer.output_fits_path = results_fits_path

    # --- Read Metadata ---
    metadata = reader_analyzer.get_fits_metadata(results_fits_path)
    if metadata is None: return
    detector = metadata.get('detector', 'unknown')
    stamp_hw = metadata.get('params', {}).get('stamp_half_width', 5)

    # --- Load Original Image Data ---
    print(f"\nLoading original image data needed for stamp extraction...")
    # Use LFC path from config
    reader_analyzer.lfc_path = lfc_file_path
    if not reader_analyzer.load_data(detector=detector):
        print("Failed to load original image data."); return

    # --- Read Peak Catalog ---
    print("\nReading peak catalog...")
    peak_catalog = reader_analyzer.read_peak_catalog(results_fits_path)
    if peak_catalog is None: return

    # --- Select Target Peak ---
    print(f"Selecting target: Order={target_order}, Image={target_image_str}, Segment={target_segment}, PeakIndex={target_peak_index}")
    segment_mask = (peak_catalog['ORDER_NUM'] == target_order) & \
                   (peak_catalog['IMGTYPE'] == img_type_int) & \
                   (peak_catalog['SEGMENT'] == target_segment)
    peaks_in_segment = peak_catalog[segment_mask]
    if len(peaks_in_segment) == 0: print("Error: No peaks found for specified target."); return
    if target_peak_index < 0 or target_peak_index >= len(peaks_in_segment): print(f"Error: peak_index out of bounds."); return
    target_peak = peaks_in_segment[target_peak_index]
    peak_x, peak_y = target_peak['PEAK_X'], target_peak['PEAK_Y']
    print(f"Target peak selected at ({peak_x}, {peak_y})")

    # --- Extract Stamp Data ---
    print(f"Extracting stamp data around ({peak_x}, {peak_y})...")
    stamp_data = reader_analyzer.get_stamp_data(peak_x, peak_y, stamp_half_width=stamp_hw)
    if stamp_data is None: print(f"Error: Could not extract stamp data."); return
    print(f"Stamp extracted with shape: {stamp_data.shape}")

    # --- Prepare Prior Information (NEEDS IMPLEMENTATION) ---
    prior_info = {'global_x': float(peak_x), 'global_y': float(peak_y)}
    # Load expected RNV, spatial models etc. from config or separate file
    prior_info['expected_rnv'] = config.get('prior_defaults', {}).get('expected_rnv', 25.0)
    print("Using placeholder/default priors for spatial variation.")

    # --- Initialize Bayesian Fitter ---
    print("\nInitializing Bayesian Stamp Fitter...")
    bayesian_fitter = BayesianStampFitter(
        n_max_zern_stamp=nmax_stamp,
        n_poly_cont=npoly_cont,
        zernike_indices_to_fit=zernike_indices_to_fit
    )

    # --- Run Fit ---
    print("\nRunning MCMC fit...")
    start_fit_time = time.time()
    mcmc = bayesian_fitter.fit(
        stamp_data=stamp_data, global_x=peak_x, global_y=peak_y, prior_info=prior_info,
        num_warmup=num_warmup, num_samples=num_samples, seed=seed, init_strategy=init_strategy
    )
    end_fit_time = time.time()
    if mcmc is None: print("MCMC fitting failed."); return
    print(f"MCMC fitting completed in {end_fit_time - start_fit_time:.2f} seconds.")

    # --- Process and Plot Results ---
    print("\n--- Fit Results ---")
    bayesian_fitter.print_summary()
    summary_dict = bayesian_fitter.get_results_summary()
    if summary_dict and 'fit_stats' in summary_dict:
        stats = summary_dict['fit_stats']; print("\nFit Statistics:");
        print(f"  Log Likelihood: {stats.get('loglik', np.nan):.3f}"); print(f"  Num Params (k): {stats.get('k', np.nan)}")
        print(f"  Num Data (n):   {stats.get('n', np.nan)}"); print(f"  Red. Chi2:      {stats.get('red_chi2', np.nan):.3f}")
        print(f"  AIC:            {stats.get('aic', np.nan):.3f}"); print(f"  AICc:           {stats.get('aicc', np.nan):.3f}")
        print(f"  BIC:            {stats.get('bic', np.nan):.3f}")

    if enable_plots:
        print("\nGenerating result plots...")
        base_filename = f"O{target_order}{target_image_str}_S{target_segment}_P{target_peak_index}"
        title_prefix = f"O={target_order}{target_image_str} S={target_segment} P={target_peak_index}"

        # --- Overview Plot ---
        fname_overview = plot_dir / f"{base_filename}_bayes_overview.{plot_format}" if save_plots else None
        try:
            bayesian_fitter.plot_fit_overview(stamp_data, filename=fname_overview, title_prefix=title_prefix)
        except Exception as e: print(f"Error plotting overview: {e}")

        # --- Other optional plots ---
        plot_post = plot_settings.get('posteriors',{}).get('enable', True)
        plot_corner = plot_settings.get('corner',{}).get('enable', True)

        if plot_post:
            fname_post = plot_dir / f"{base_filename}_bayes_posteriors.{plot_format}" if save_plots else None
            # Parameters to plot: Include basics and the Zernike coefficient vector
            params_post = ['bias', 'A_line', 'A_cont', 'dx', 'dy', 'read_noise_var']
            if bayesian_fitter.n_zernikes_fitted > 0:
                # Plot the final derived coefficients, not the offsets
                params_post.append('zernike_coeffs_final')
            try:
                bayesian_fitter.plot_posteriors(params_to_plot=params_post, filename=fname_post)
            except Exception as e: print(f"Error plotting posteriors: {e}")

        if plot_corner:
            fname_corner = plot_dir / f"{base_filename}_bayes_corner.{plot_format}" if save_plots else None
            # Select a *small subset* for corner plot (including maybe first Zernike coeff)
            params_corner = ['log_A_line', 'log_bias_offset', 'dx', 'dy', 'log_read_noise_var']
            if bayesian_fitter.n_zernikes_fitted > 0:
                 # Get the actual derived coeff vector and select first element(s)
                 # Note: Corner plot needs individual param names if plotting multiple Zernikes
                 # It's easier to plot the log_offset for the first Zernike here
                 params_corner.append('z_offset[0]') # Plot offset for Z[0]
                 if bayesian_fitter.n_zernikes_fitted > 1:
                      params_corner.append('z_offset[1]') # Plot offset for Z[1]
            try:
                bayesian_fitter.plot_corner(params_to_plot=params_corner, filename=fname_corner)
            except Exception as e: print(f"Error plotting corner: {e}")

    print("\nSingle stamp fitting finished.")


if __name__ == "__main__":
    main()