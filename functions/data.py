#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:27:34 2023

@author: dmilakov
"""

import numpy as np
import harps.functions.spectral as specfunc
import harps.background as bkg

def prepare_data1d(wav1d_in, flx1d_in, bkg1d_in=None, redisperse=False, subbkg=True,
                   velocity_step=0.82, plot=False):
    """
    Prepares 1D spectral data for analysis.

    Handles background subtraction and error calculation, with an option
    for redispersion. Ensures errors are finite.

    Parameters:
    -----------
    wav1d_in : np.ndarray
        Input 1D wavelength array.
    flx1d_in : np.ndarray
        Input 1D flux array.
    bkg1d_in : np.ndarray, optional
        Input 1D background array. If None and subbkg is True,
        an error will be raised (or could be calculated if a method was provided).
        Default is None.
    redisperse : bool, optional
        If True, redisperse the spectrum. Default is False.
    subbkg : bool, optional
        If True, subtract background. Default is True.
    velocity_step : float, optional
        Velocity step for redispersion (km/s). Default is 0.82.
    plot : bool, optional
        If True, generate plots (currently not implemented). Default is False.

    Returns:
    --------
    wav1d : np.ndarray
        Processed 1D wavelength array.
    flx1d : np.ndarray
        Processed 1D flux array (background subtracted if specified).
    err1d : np.ndarray
        Calculated 1D error array.
    """
    
    assert len(flx1d_in)==len(wav1d_in)
    
    wav1d = wav1d_in.copy()
    flx1d = flx1d_in.copy()
    # Initial error for the flux, ensuring non-negativity for sqrt
    # This assumes Poisson noise where variance ~ flux.
    err1d = np.sqrt(np.maximum(1, flx1d))
    
    bkg1d = None
    err_bkg1d = None
    if bkg1d_in is not None:
        bkg1d = bkg1d_in.copy()
        # Initial error for the background, ensuring non-negativity for sqrt
        err_bkg1d = np.sqrt(np.maximum(0., bkg1d))
        
    if redisperse:
        # Redisperse flux and its error
        # wav1d might be modified by redisperse1d if it creates a new grid
        wav1d, flx1d, err1d = specfunc.redisperse1d(wav1d, flx1d, err1d, velocity_step)
        
        if bkg1d is not None:
            # Redisperse background and its error to the NEW wavelength grid
            _, bkg1d, err_bkg1d = specfunc.redisperse1d(wav1d, bkg1d, err_bkg1d, velocity_step)
            # We use the new wav1d from flux redispersion to ensure consistency.
            # The first returned value (wavelength array for background) is ignored
            # as it should match the new wav1d.
    if subbkg:
        if bkg1d is not None:
            flx1d_gross = flx1d.copy() # Flux before subtraction (potentially redispersed)
            bkg1d_to_subtract = bkg1d.copy() # Background (potentially redispersed)

            flx1d = flx1d_gross - bkg1d_to_subtract
            
            # Error propagation for subtraction: Var(Net) = Var(Gross) + Var(Bkg)
            # If using propagated errors from redispersion (which should be stddevs):
            if redisperse and err_bkg1d is not None:
                 # err1d and err_bkg1d are stddevs from redispersion
                err1d = np.sqrt(err1d**2 + err_bkg1d**2)
            else:
                # If not redispersed, or if bkg error wasn't tracked via redispersion,
                # calculate from current flux/bkg values assuming Poisson noise.
                var_gross = np.maximum(1., np.abs(flx1d_gross))
                var_bkg = np.maximum(1., np.abs(bkg1d_to_subtract))
                err1d = np.sqrt(var_gross + var_bkg)
        else:
            # Background subtraction is requested, but no background provided.
            # Original code had a commented-out call to bkg.get_env_bkg.
            # You might want to re-enable or modify this.
            # For now, raising an error is safer.
            # Example:
            # env1d, calculated_bkg1d = bkg.get_env_bkg(flx1d, extrema1d, xarray=None, yerror=err1d,
            #                                           kind='fit_spline')
            # flx1d = flx1d - calculated_bkg1d
            # err1d = np.sqrt(np.maximum(0., flx1d + calculated_bkg1d) + np.maximum(0., calculated_bkg1d))
            raise ValueError("Background subtraction (subbkg=True) requested, "
                             "but no background (bkg1d) was provided or calculated.")
    # If not subbkg, err1d is already the error of flx1d (potentially redispersed).
    # This error was calculated as sqrt(max(0, flx1d)) or propagated by redisperse.
    
    # Final check to ensure all error values are finite
    if not np.all(np.isfinite(err1d)):
        # This might happen if flx1d or bkg1d contained NaNs or Infs initially
        # Or if specfunc.redisperse1d introduced non-finite values
        num_non_finite = np.sum(~np.isfinite(err1d))
        # print(f"Warning: err1d contains {num_non_finite} non-finite values. Replacing with large number or NaN.")
        err1d = np.nan_to_num(err1d, nan=np.nan, posinf=np.finfo(err1d.dtype).max, neginf=np.finfo(err1d.dtype).max)
        # Or, more commonly, replace with NaN and let downstream handle it:
        # err1d[~np.isfinite(err1d)] = 2**32


    return wav1d, flx1d, err1d

def prepare_data2d(wav2d_in, flx2d_in, bkg2d_in=None, redisperse=False, subbkg=True,
                   velocity_step=0.82, plot=False, sOrder=39, eOrder=None, extrema2d_arg=None): # plot unused
    """
    Prepares 2D spectral data for analysis.

    Handles background subtraction and error calculation, with an option
    for redispersion. Ensures errors are finite.

    Parameters:
    -----------
    wav2d_in : np.ndarray
        Input 2D wavelength array (n_orders, n_pixels).
    flx2d_in : np.ndarray
        Input 2D flux array (n_orders, n_pixels).
    bkg2d_in : np.ndarray, optional
        Input 2D background array. If None and subbkg is True,
        it will be calculated using bkg.get_env_bkg2d_from_array.
        Default is None.
    redisperse : bool, optional
        If True, redisperse the spectrum. Default is False.
    subbkg : bool, optional
        If True, subtract background. Default is True.
    velocity_step : float, optional
        Velocity step for redispersion (km/s). Default is 0.82.
    plot : bool, optional
        If True, generate plots (currently not implemented). Default is False.
    sOrder, eOrder : int, optional
        Start and end orders for background calculation if bkg2d_in is None.
    extrema2d_arg : any, optional
        The 'extrema2d' argument required by bkg.get_env_bkg2d_from_array.
        Renamed to avoid conflict if 'extrema2d' is a global.

    Returns:
    --------
    wav2d : np.ndarray
        Processed 2D wavelength array.
    flx2d : np.ndarray
        Processed 2D flux array (background subtracted if specified).
    err2d : np.ndarray
        Calculated 2D error array.
    """
    if not isinstance(flx2d_in, np.ndarray) or not isinstance(wav2d_in, np.ndarray):
        raise TypeError("flx2d_in and wav2d_in must be numpy arrays.")
    if flx2d_in.shape != wav2d_in.shape:
        raise ValueError("flx2d_in and wav2d_in must have the same shape.")
    if bkg2d_in is not None and not isinstance(bkg2d_in, np.ndarray):
        raise TypeError("bkg2d_in, if provided, must be a numpy array.")
    if bkg2d_in is not None and bkg2d_in.shape != flx2d_in.shape:
        raise ValueError("bkg2d_in, if provided, must have the same shape as flx2d_in.")

    wav2d = wav2d_in.copy()
    flx2d = flx2d_in.copy()

    # Initial error for the flux, ensuring non-negativity for sqrt
    err2d = np.sqrt(np.maximum(0., flx2d))

    # Handle background array
    bkg2d = None
    err_bkg2d = None
    if bkg2d_in is not None:
        bkg2d = bkg2d_in.copy()
        err_bkg2d = np.sqrt(np.maximum(0., bkg2d))

    if redisperse:
        # wav2d might be modified by redisperse2d
        wav2d, flx2d, err2d = specfunc.redisperse2d(wav2d, flx2d, err2d, velocity_step)
        
        if bkg2d is not None: # If background was provided
            _, bkg2d, err_bkg2d = specfunc.redisperse2d(wav2d, bkg2d, err_bkg2d, velocity_step)
            # Background is redispersed to the new grid defined by wav2d from flux redispersion.

    if subbkg:
        if bkg2d is None: # Background not provided, calculate it
            if extrema2d_arg is None: # extrema2d was used from global scope in original
                raise ValueError("extrema2d_arg must be provided for background calculation.")
            print("Calculating 2D background as it was not provided...") # Optional: for verbosity
            # Ensure flx2d passed to background estimation is the current state (e.g. after redispersion)
            # The error (err2d at this point) might be useful for some background algorithms
            env2d, bkg2d_calculated = bkg.get_env_bkg2d_from_array(
                flx2d,
                extrema2d_arg, # Use the passed argument
                sOrder=sOrder,
                eOrder=eOrder,
                kind='fit_spline'
                # consider passing yerror=err2d if your function uses it
            )
            bkg2d = bkg2d_calculated
            # Error for the calculated background
            err_bkg2d = np.sqrt(np.maximum(0., bkg2d))
            # If calculated background was also redispersed (if original flx wasn't and redisperse=True):
            # This part can get complex. Simpler if background is calculated on already-redispersed flux.
            # If redisperse was True, flx2d is already redispersed. So bkg2d is calculated on that grid.
            # If redisperse was False, flx2d is on original grid.
            # No further redispersion of bkg2d needed here as it's derived from current flx2d grid.

        # Now, bkg2d should exist (either provided or calculated).
        if bkg2d is None: # Should not happen if logic above is correct
             raise RuntimeError("Background is still None after attempting to provide or calculate it.")

        flx2d_gross = flx2d.copy()
        bkg2d_to_subtract = bkg2d.copy()

        flx2d = flx2d_gross - bkg2d_to_subtract
        
        # Error propagation: Var(Net) = Var(Gross) + Var(Bkg)
        if redisperse and err_bkg2d is not None and bkg2d_in is not None: # i.e. bkg was provided and redispersed
             # err2d and err_bkg2d are stddevs from redispersion
            err2d = np.sqrt(err2d**2 + err_bkg2d**2)
        else:
            # If not redispersed, or if bkg was calculated (so err_bkg2d is fresh),
            # or if bkg_error from redispersion wasn't available.
            var_gross = np.maximum(0., np.abs(flx2d_gross))
            var_bkg = np.maximum(0., np.abs(bkg2d_to_subtract))
            err2d = np.sqrt(var_gross + var_bkg)
            
    # Final check for non-finite errors
    if not np.all(np.isfinite(err2d)):
        num_non_finite = np.sum(~np.isfinite(err2d))
        # print(f"Warning: err2d contains {num_non_finite} non-finite values. Replacing with NaN.")
        err2d[~np.isfinite(err2d)] = np.nan


    return wav2d, flx2d, err2d

    
    
    
        
    
    
    
    