#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:27:15 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
# import zernpy # Install using: pip install zernpy
import warnings
import traceback

try:
    import zernpy
    ZERNPY_AVAILABLE = True
except ImportError:
    warnings.warn("zernpy library not found. Zernike fitting requires 'pip install zernpy'")
    ZERNPY_AVAILABLE = False

# --- Utility Functions (Can be kept separate or inside the class if preferred) ---

def generate_zernike_indices(n_max):
    """Generates a list of (n, m) Zernike indices up to radial order n_max."""
    indices = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            indices.append((n, m))
    return indices

# --- Zernike Fitter Class ---

class ZernikeFitter:
    """
    Fits 3D data (X, Y, Z) using Zernike polynomials with centroid optimization.

    Attributes:
        n_max (int): Maximum radial order for Zernike polynomials.
        r_max (float): Radius used for normalizing coordinates (data outside
                       this radius, after centering, is effectively ignored
                       by Zernike calculation).
        zernike_indices (list): List of (n, m) Zernike indices used for fitting.
        n_coeffs (int): Number of Zernike coefficients.
        fitted_params (np.ndarray | None): Optimal parameters [xc, yc, c0, c1,...]
                                           from the fit, or None if fit failed.
        param_errors (np.ndarray | None): Standard errors of the fitted parameters,
                                          or None if fit failed or covariance
                                          could not be estimated.
        fitted_xc (float | None): Fitted X centroid.
        fitted_yc (float | None): Fitted Y centroid.
        fitted_coeffs (np.ndarray | None): Fitted Zernike coefficients.
        err_xc (float | None): Standard error of fitted X centroid.
        err_yc (float | None): Standard error of fitted Y centroid.
        err_coeffs (np.ndarray | None): Standard errors of fitted Zernike coefficients.
        rmse (float | None): Root Mean Squared Error of the fit.
        r_squared (float | None): R-squared value of the fit.
        success (bool): Whether the curve_fit converged successfully.
        message (str): Status message from the fitting process.
    """
    def __init__(self, n_max, r_max):
        """
        Initializes the ZernikeFitter.

        Args:
            n_max (int): Maximum radial order for Zernike polynomials.
            r_max (float): Radius for normalizing coordinates before applying
                           Zernike polynomials. Data should ideally be roughly
                           centered and scaled such that the features of interest
                           fall within this radius.
        """
        if not isinstance(n_max, int) or n_max < 0:
            raise ValueError("n_max must be a non-negative integer.")
        if not isinstance(r_max, (int, float)) or r_max <= 0:
             raise ValueError("r_max must be a positive number.")

        self.n_max = n_max
        self.r_max = r_max
        self.zernike_indices = generate_zernike_indices(self.n_max)
        self.n_coeffs = len(self.zernike_indices)

        # Initialize result attributes
        self.fitted_params = None
        self.param_errors = None
        self.fitted_xc = None
        self.fitted_yc = None
        self.fitted_coeffs = None
        self.err_xc = None
        self.err_yc = None
        self.err_coeffs = None
        self.rmse = None
        self.r_squared = None
        self.pcov = None # Store covariance matrix
        self.success = False
        self.message = "Fit not yet performed."

    def _calculate_zernike_basis(self, x_norm, y_norm):
        """
        Internal method: Calculates the Zernike basis matrix for normalized coordinates.

        Uses zernpy's polynomial_value which expects polar coordinates,
        handling the conversion internally. Points outside the unit disk (rho > 1)
        will have basis values of 0.

        Args:
            x_norm (np.ndarray): Normalized X coordinates (-1 to 1 range expected
                                 for points within r_max).
            y_norm (np.ndarray): Normalized Y coordinates (-1 to 1 range expected
                                 for points within r_max).

        Returns:
            np.ndarray: Basis matrix (n_points, n_zernikes).
        """
        if x_norm is None or y_norm is None or len(x_norm) != len(y_norm):
            raise ValueError("Invalid or mismatched x_norm/y_norm arrays.")
        if len(x_norm) == 0:
             # Return an empty array with the correct number of columns
             return np.zeros((0, self.n_coeffs))

        n_points = len(x_norm)
        basis_matrix = np.zeros((n_points, self.n_coeffs))

        # --- Convert normalized cartesian to polar ---
        rho = np.sqrt(x_norm**2 + y_norm**2)
        # Use arctan2 for correct angle in all quadrants
        theta = np.arctan2(y_norm, x_norm)

        # --- Create Mask for points within the unit disk ---
        # Zernikes are defined within rho <= 1
        valid_mask = rho <= 1.0

        # If no points are within the unit disk, return the zero matrix
        if not np.any(valid_mask):
            # This might happen if r_max is chosen too small or data is shifted way off
            # warnings.warn("No points found within the unit disk (rho <= 1) after normalization.")
            return basis_matrix # Return the matrix of zeros

        # --- Extract valid polar coordinates ---
        rho_valid = rho[valid_mask]
        theta_valid = theta[valid_mask]

        # Create a temporary basis matrix only for the valid points
        temp_basis_valid = np.zeros((len(rho_valid), self.n_coeffs))

        for i, (n, m) in enumerate(self.zernike_indices):
            try:
                zp = zernpy.ZernPol(m=m, n=n) # zernpy uses (m, n) order
                temp_basis_valid[:, i] = zp.polynomial_value(rho_valid, theta_valid)
            except Exception as e:
                warnings.warn(f"Error calculating Zernike (n={n}, m={m}) with zernpy: {e}. Filling with zeros.")
                # temp_basis_valid[:, i] = np.nan # Or 0, depending on how you want to handle errors
                temp_basis_valid[:, i] = 0

        # Place the calculated values for valid points back into the full basis matrix
        basis_matrix[valid_mask, :] = temp_basis_valid

        return basis_matrix

    def _zernike_model_with_centroid(self, xy_data_orig, x_c, y_c, *coeffs_tuple):
        """
        Internal model function for curve_fit. Uses instance attributes
        self.r_max and self.zernike_indices.
        """
        X_orig = xy_data_orig[0]
        Y_orig = xy_data_orig[1]
        coeffs = np.array(coeffs_tuple) # Convert tuple back to array

        if len(coeffs) != self.n_coeffs:
            # This check is important as curve_fit passes coeffs as *args
            raise ValueError(f"Internal error: Number of coefficients ({len(coeffs)}) "
                             f"does not match fitter's expected number ({self.n_coeffs})")

        # --- Coordinate Transformation ---
        X_shifted = X_orig - x_c
        Y_shifted = Y_orig - y_c
        # Normalize using the instance's r_max
        X_norm = X_shifted / self.r_max
        Y_norm = Y_shifted / self.r_max

        # --- Calculate Zernike Basis ---
        # Use the internal method which uses self.zernike_indices
        basis_matrix = self._calculate_zernike_basis(X_norm, Y_norm)

        # --- Predict Z ---
        # Matrix multiplication: (n_points, n_coeffs) @ (n_coeffs,) -> (n_points,)
        Z_pred = basis_matrix @ coeffs
        return Z_pred.ravel() # Ensure it's a 1D array for curve_fit


    def fit(self, x_orig, y_orig, z_orig, initial_guess=None, bounds=None, maxfev=10000, verbose=True):
        """
        Fits the Zernike model to the provided data.

        Args:
            x_orig (np.ndarray): 1D array of original X coordinates.
            y_orig (np.ndarray): 1D array of original Y coordinates.
            z_orig (np.ndarray): 1D array of corresponding Z values.
            initial_guess (dict | None): Optional dictionary with initial guesses for
                                         'xc', 'yc', and 'coeffs' (list or array).
                                         If None, defaults are generated.
            bounds (dict | None): Optional dictionary with bounds for 'xc', 'yc',
                                  and 'coeffs'. Keys 'lower' and 'upper' should
                                  contain lists/arrays: [xc, yc, c0, c1,...].
                                  If None, default bounds are used.
            maxfev (int): Maximum number of function evaluations for curve_fit.
            verbose (bool): If True, print status messages during fitting.

        Returns:
            bool: True if the fit converged successfully, False otherwise.
        """
        self.success = False # Reset status
        self.message = "Starting fit..."
        if verbose: print(self.message)

        if not (len(x_orig) == len(y_orig) == len(z_orig)):
            self.message = "Error: Input arrays X, Y, Z must have the same length."
            if verbose: print(self.message)
            return False
        if len(x_orig) < self.n_coeffs + 2:
             self.message = (f"Error: Number of data points ({len(x_orig)}) is less than "
                             f"the number of parameters ({self.n_coeffs + 2}). Fit is underdetermined.")
             if verbose: print(self.message)
             return False


        xy_data_fit = np.vstack((x_orig.ravel(), y_orig.ravel()))
        z_data_fit = z_orig.ravel()

        # --- Sensible Default Initial Guesses ---
        p0 = [0.0, 0.0] + [0.0] * self.n_coeffs
        # Estimate initial piston term (Z(0,0)) if present
        try:
            piston_index = self.zernike_indices.index((0, 0))
            # Initial piston guess based on mean Z, but only if data exists
            if len(z_data_fit) > 0:
                 p0[2 + piston_index] = np.mean(z_data_fit)
        except ValueError:
            piston_index = -1 # Piston term not included in fit

        # --- Override defaults if initial_guess is provided ---
        if initial_guess is not None:
            if 'xc' in initial_guess: p0[0] = initial_guess['xc']
            if 'yc' in initial_guess: p0[1] = initial_guess['yc']
            if 'coeffs' in initial_guess:
                if len(initial_guess['coeffs']) == self.n_coeffs:
                    p0[2:] = list(initial_guess['coeffs'])
                else:
                    warnings.warn(f"Length of initial 'coeffs' guess ({len(initial_guess['coeffs'])})"
                                  f" does not match n_coeffs ({self.n_coeffs}). Using default coeffs.")

        # --- Sensible Default Bounds ---
        # Allow centroid to shift within +/- r_max/2 (can be adjusted)
        # Coefficients can be anything by default
        default_bounds_lower = [-self.r_max, -self.r_max] + [-np.inf] * self.n_coeffs
        default_bounds_upper = [ self.r_max,  self.r_max] + [ np.inf] * self.n_coeffs
        fit_bounds = (default_bounds_lower, default_bounds_upper)

        # --- Override defaults if bounds are provided ---
        if bounds is not None:
             try:
                 lower = list(bounds['lower'])
                 upper = list(bounds['upper'])
                 if len(lower) == self.n_coeffs + 2 and len(upper) == self.n_coeffs + 2:
                     fit_bounds = (lower, upper)
                 else:
                     warnings.warn("Length of provided 'bounds' does not match number of parameters "
                                   f"({self.n_coeffs + 2}). Using default bounds.")
             except (KeyError, TypeError, ValueError) as e:
                 warnings.warn(f"Invalid 'bounds' dictionary format: {e}. Using default bounds.")

        if verbose:
            print(f"Fitting with n_max={self.n_max}, r_max={self.r_max}")
            print(f"Number of Zernike coefficients: {self.n_coeffs}")
            # print(f"Indices: {self.zernike_indices}") # Can be long
            print(f"Initial guess (p0): Xc={p0[0]:.3f}, Yc={p0[1]:.3f}, Coeffs=[{', '.join(f'{c:.3f}' for c in p0[2:])}]")
            # print(f"Bounds: {fit_bounds}") # Can be long

        # --- Run curve_fit ---
        try:
            popt, pcov = curve_fit(
                self._zernike_model_with_centroid,
                xy_data_fit,
                z_data_fit,
                p0=p0,
                bounds=fit_bounds,
                maxfev=maxfev,
                # sigma=None, # Add weights here if you have Z uncertainties
                # absolute_sigma=False,
            )

            self.fitted_params = popt
            self.pcov = pcov
            self.fitted_xc = popt[0]
            self.fitted_yc = popt[1]
            self.fitted_coeffs = popt[2:]

            # --- Calculate Standard Errors (if possible) ---
            if pcov is not None and np.all(np.isfinite(pcov)):
                 # Check for negative variance (bad fit or model)
                diag_pcov = np.diag(pcov)
                if np.any(diag_pcov < 0):
                     warnings.warn("Negative variance found in covariance matrix. Errors are unreliable.")
                     # Set errors to NaN or None where variance is negative
                     diag_pcov[diag_pcov < 0] = np.nan
                     self.param_errors = np.sqrt(diag_pcov)
                else:
                    self.param_errors = np.sqrt(diag_pcov)

                self.err_xc = self.param_errors[0]
                self.err_yc = self.param_errors[1]
                self.err_coeffs = self.param_errors[2:]
            else:
                 warnings.warn("Could not estimate parameter errors (pcov is None or contains non-finite values).")
                 self.param_errors = np.full_like(popt, np.nan) # Or None
                 self.err_xc = np.nan
                 self.err_yc = np.nan
                 self.err_coeffs = np.full(self.n_coeffs, np.nan)


            # --- Evaluate Fit Quality ---
            Z_pred_final = self._zernike_model_with_centroid(xy_data_fit, *self.fitted_params)
            residuals = z_data_fit - Z_pred_final
            self.rmse = np.sqrt(np.mean(residuals**2))

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((z_data_fit - np.mean(z_data_fit))**2)
            if ss_tot > 1e-12: # Avoid division by zero for flat data
                self.r_squared = 1 - (ss_res / ss_tot)
            elif ss_res < 1e-12: # If residuals are also tiny, it's a perfect fit to flat data
                self.r_squared = 1.0
            else: # If residuals exist but data variation is zero, R^2 is undefined or negative infinity
                 self.r_squared = -np.inf # Or np.nan or 0, depending on convention


            self.success = True
            self.message = "curve_fit completed successfully."
            if verbose: print(self.message)


        except RuntimeError as e:
            self.message = f"Error during curve_fit: {e}. Fit failed to converge."
            if verbose: print(self.message)
            # Keep results as None
        except ValueError as e:
             self.message = f"ValueError during setup or fitting: {e}."
             if verbose: print(self.message)
             # traceback.print_exc() # Optional: for debugging value errors
        except ImportError:
             self.message = "ImportError: Ensure 'zernpy' is installed (`pip install zernpy`)"
             if verbose: print(self.message)
             raise # Re-raise import error as it's fundamental
        except Exception as e:
             self.message = f"An unexpected error occurred during fitting: {e}"
             if verbose:
                 print(self.message)
                 traceback.print_exc()
             # Keep results as None

        return self.success

    def predict(self, x_orig, y_orig):
        """
        Predicts Z values using the fitted model for given X, Y coordinates.

        Args:
            x_orig (np.ndarray): 1D array of X coordinates.
            y_orig (np.ndarray): 1D array of Y coordinates.

        Returns:
            np.ndarray | None: Predicted Z values, or None if the model hasn't been
                               successfully fitted yet.
        """
        if not self.success or self.fitted_params is None:
            print("Warning: Model has not been successfully fitted yet. Cannot predict.")
            return None

        if len(x_orig) != len(y_orig):
             raise ValueError("Input arrays X and Y must have the same length.")

        xy_data_pred = np.vstack((x_orig.ravel(), y_orig.ravel()))
        z_pred = self._zernike_model_with_centroid(xy_data_pred, *self.fitted_params)
        return z_pred

    def get_results(self, include_coeffs_table=True):
        """
        Returns a dictionary containing the fit results.

        Args:
             include_coeffs_table (bool): If True, includes a formatted string
                                         list of coefficients and errors.

        Returns:
            dict: A dictionary containing fit parameters, errors, and metrics.
                  Returns a basic status dict if fit hasn't run or failed.
        """
        if not self.success or self.fitted_params is None:
            return {
                "success": self.success,
                "message": self.message,
            }

        results = {
            "success": self.success,
            "message": self.message,
            "n_max": self.n_max,
            "r_max": self.r_max,
            "fitted_xc": self.fitted_xc,
            "fitted_yc": self.fitted_yc,
            "err_xc": self.err_xc,
            "err_yc": self.err_yc,
            "fitted_coeffs": self.fitted_coeffs,
            "err_coeffs": self.err_coeffs,
            "rmse": self.rmse,
            "r_squared": self.r_squared,
            "zernike_indices": self.zernike_indices,
            # Optional: include full params array and cov matrix if needed
            # "fitted_params": self.fitted_params,
            # "covariance_matrix": self.pcov,
        }

        if include_coeffs_table:
            coeffs_lines = []
            coeffs_lines.append("Fitted Zernike Coefficients:")
            for i, (n, m) in enumerate(self.zernike_indices):
                 coeff_val = self.fitted_coeffs[i]
                 err_val = self.err_coeffs[i] if self.err_coeffs is not None and np.isfinite(self.err_coeffs[i]) else np.nan
                 coeffs_lines.append(f"  Z(n={n}, m={m}): {coeff_val:.5f} +/- {err_val:.5f}")
            results["coefficients_table"] = "\n".join(coeffs_lines)

        return results

    def plot_fit_comparison(self, x_orig, y_orig, z_orig, n_grid=50, title=None, filename=None):
        """
        Generates a 3D plot comparing the original data and the fitted surface.

        Args:
            x_orig (np.ndarray): Original X data.
            y_orig (np.ndarray): Original Y data.
            z_orig (np.ndarray): Original Z data.
            n_grid (int): Resolution of the grid for plotting the fitted surface.
            title (str | None): Optional title for the plot. If None, a default
                                title is generated.
            filename (str | None): Optional path to save the figure. If None,
                                   the plot is shown interactively.
        """
        if not self.success or self.fitted_params is None:
            print("Warning: Model has not been successfully fitted yet. Cannot plot.")
            return

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of original data
        ax.scatter(x_orig, y_orig, z_orig, c='blue', s=5, alpha=0.2, label='Original Data')

        # Create grid for plotting the fitted surface
        # Determine grid range based on original data or fitted center +/- r_max
        # Using fitted center and r_max is often better for visualizing the fit region
        x_min_plot = self.fitted_xc - self.r_max * 1.1
        x_max_plot = self.fitted_xc + self.r_max * 1.1
        y_min_plot = self.fitted_yc - self.r_max * 1.1
        y_max_plot = self.fitted_yc + self.r_max * 1.1

        # # Alternative: Use original data range
        # x_min_plot = np.min(x_orig)
        # x_max_plot = np.max(x_orig)
        # y_min_plot = np.min(y_orig)
        # y_max_plot = np.max(y_orig)


        x_grid_orig = np.linspace(x_min_plot, x_max_plot, n_grid)
        y_grid_orig = np.linspace(y_min_plot, y_max_plot, n_grid)
        X_grid_orig, Y_grid_orig = np.meshgrid(x_grid_orig, y_grid_orig)

        # Predict Z values on the grid using the fitted model
        Z_grid_pred = self.predict(X_grid_orig.ravel(), Y_grid_orig.ravel())

        if Z_grid_pred is not None:
            Z_grid_pred = Z_grid_pred.reshape(X_grid_orig.shape)
            surf = ax.plot_surface(X_grid_orig, Y_grid_orig, Z_grid_pred, cmap='viridis', alpha=0.7, label='Fitted Surface')
             # Fake proxy artist for surface legend (surfaces don't automatically get legend handles)
            surf._facecolors2d=surf._facecolor3d
            surf._edgecolors2d=surf._edgecolor3d

        ax.set_xlabel('X original')
        ax.set_ylabel('Y original')
        ax.set_zlabel('Z')

        if title is None:
            title = f'Original Data and Zernike Fit (n_max={self.n_max}, R_max={self.r_max})'
        ax.set_title(title)
        ax.legend()

        if filename:
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
            plt.close(fig) # Close the figure after saving
        else:
            plt.show()
        # Add this method inside the ZernikeFitter class definition

    def plot_fit_residuals(self, x_orig, y_orig, z_orig, title=None, filename=None):
        """
        Generates a 3D scatter plot of the residuals (z_orig - z_pred)
        at the original data point locations.

        Args:
            x_orig (np.ndarray): Original X data (1D array).
            y_orig (np.ndarray): Original Y data (1D array).
            z_orig (np.ndarray): Original Z data (1D array).
            title (str | None): Optional title for the plot. If None, a default
                                title is generated.
            filename (str | None): Optional path to save the figure. If None,
                                   the plot is shown interactively.
        """
        # --- Input Checks ---
        if not self.success or self.fitted_params is None:
            print("Warning: Model has not been successfully fitted yet. Cannot plot residuals.")
            return
        if not isinstance(x_orig, np.ndarray) or not isinstance(y_orig, np.ndarray) or not isinstance(z_orig, np.ndarray):
             print("Warning: x_orig, y_orig, z_orig must be NumPy arrays.")
             return
        if not (x_orig.shape == y_orig.shape == z_orig.shape):
              print("Warning: x_orig, y_orig, z_orig must have the same shape.")
              return
        if not (len(x_orig) == len(y_orig) == len(z_orig)):
             print("Warning: Input arrays X, Y, Z must have the same length. Cannot plot residuals.")
             return
        if len(x_orig) == 0:
            print("Warning: Input data is empty. Cannot plot residuals.")
            return

        # --- Calculate Predictions and Residuals ---
        # Predict Z values at the original data points using the fitted model
        z_pred = self.predict(x_orig.ravel(), y_orig.ravel()) # predict method handles raveling internally

        if z_pred is None:
             # This might happen if predict fails for some reason after fit succeeded
             print("Warning: Could not predict Z values for residual calculation.")
             return

        # Calculate residuals
        residuals = z_orig.ravel() - z_pred # Both should be 1D arrays of the same size now
        # --- Determine Symmetric Color Limits ---
        max_abs_res = np.nanmax(np.abs(residuals)) if len(residuals) > 0 else 0
        if not np.isfinite(max_abs_res) or max_abs_res < 1e-9:
            # Handle cases with no residuals, all zero, or NaN residuals
            # Set a default small range if needed, otherwise scatter might fail
            color_vmin, color_vmax = -1e-9, 1e-9
            # Or optionally print a warning and use automatic scaling (but it won't be symmetric)
            # print("Warning: Residuals are zero or non-finite; using automatic color scaling.")
            # color_vmin, color_vmax = None, None
        else:
            # Set symmetric limits around zero
            color_limit = max_abs_res
            color_vmin = -color_limit
            color_vmax = color_limit
        
        # --- Plotting ---
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of residuals, colored by residual value for better visualization
        sc = ax.scatter(x_orig, y_orig, residuals, 
                        c=residuals, cmap='coolwarm', 
                        vmin=color_vmin, vmax=color_vmax,
                        s=5, alpha=0.6)
        fig.colorbar(sc, label='Residual Value (Z_orig - Z_pred)')

        # Add a reference plane at z=0 (perfect fit) for context
        # Determine plane bounds from data range to ensure it covers the data extent
        x_min_plot, x_max_plot = np.min(x_orig), np.max(x_orig)
        y_min_plot, y_max_plot = np.min(y_orig), np.max(y_orig)
        # Handle edge case where data might be perfectly aligned along one axis
        if x_max_plot == x_min_plot: x_max_plot += 1 # Add small width if X is constant
        if y_max_plot == y_min_plot: y_max_plot += 1 # Add small depth if Y is constant

        xx, yy = np.meshgrid(np.linspace(x_min_plot, x_max_plot, 2),
                             np.linspace(y_min_plot, y_max_plot, 2))
        zz = np.zeros_like(xx) # Plane at z=0 (residual = 0)
        # Plot the zero plane with low opacity
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', label='Zero Residual Plane', linewidth=0, antialiased=False)


        # --- Labels and Title ---
        ax.set_xlabel('X original')
        ax.set_ylabel('Y original')
        ax.set_zlabel('Residuals (Z_orig - Z_pred)')

        if title is None:
            # Generate a default title if none provided
            title = f'Fit Residuals (n_max={self.n_max}, R_max={self.r_max})'
        ax.set_title(title)
        # ax.legend() # Legend for the plane might be useful, but scatter points have no single label

        # --- Adjust Z limits ---
        # Optionally center the Z (residual) axis around 0 for better visualization
        max_abs_res = np.max(np.abs(residuals))
        if np.isfinite(max_abs_res) and max_abs_res > 1e-9: # Avoid issues with tiny/zero residuals
             ax.set_zlim(-max_abs_res * 1.1, max_abs_res * 1.1) # Symmetrical limits


        # --- Show or Save ---
        if filename:
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Residual plot saved to {filename}")
            except Exception as e:
                print(f"Error saving residual plot to {filename}: {e}")
            finally:
                plt.close(fig) # Close the figure after attempting to save
        else:
            plt.show() # Display the plot interactively
    def plot_power_spectrum(self, title=None, filename=None, plot_type='abs'):
        """
        Plots the Zernike coefficient spectrum (magnitude or squared value)
        based on the results stored in this fitter instance.

        Args:
            title (str, optional): Title for the plot. If None, a default title is generated.
            filename (str or Path, optional): Path to save the figure. If None, shows interactively.
            plot_type (str): Type of value to plot: 'coeff' (raw value), 'abs' (absolute value),
                             'sq' (squared value - "power"). Defaults to 'abs'.
        """
        # --- Check if fit results are available ---
        if not self.success or self.fitted_coeffs is None:
            print("Warning: Fit not successful or no coefficients available. Cannot plot power spectrum.")
            return
        if self.zernike_indices is None:
            print("Warning: Zernike indices not available. Cannot plot power spectrum.")
            return

        # --- Get data from instance attributes ---
        coefficients = self.fitted_coeffs
        errors = self.err_coeffs # This might be None or contain NaNs if error calculation failed
        indices = self.zernike_indices

        n_coeffs = len(coefficients)
        l_indices = np.arange(n_coeffs) # Sequential index

        # --- Determine values and errors to plot based on plot_type ---
        plot_errors_valid = False # Flag to track if errors can be plotted
        if errors is not None and len(errors) == n_coeffs and np.all(np.isfinite(errors)):
             plot_errors_valid = True
             error_values = errors # Start with original errors
        else:
             error_values = np.zeros(n_coeffs) # Use zeros if invalid/missing


        if plot_type == 'coeff':
            values_to_plot = coefficients
            y_label = "Coefficient Value"
            # Use original error_values
        elif plot_type == 'sq':
            values_to_plot = coefficients**2
            y_label = "Coefficient Squared (Power)"
            if plot_errors_valid:
                # Error propagation for square: err(x^2) approx |2*x*err(x)|
                error_values = np.abs(2 * coefficients * error_values)
            else:
                plot_errors_valid = False # Cannot propagate if original invalid
        else: # Default to 'abs'
            values_to_plot = np.abs(coefficients)
            y_label = "Coefficient Absolute Value"
            if plot_errors_valid:
                # Error for abs(x) is approx err(x) if x != 0
                # Mask error where coefficient is near zero (error bar doesn't make sense)
                zero_coeff_mask = np.abs(coefficients) < 1e-9
                error_values[zero_coeff_mask] = 0 # Set error to 0 where coeff is effectively 0
            # Keep plot_errors_valid as it was (depends only on original errors being valid)


        # --- Create Plot ---
        fig, ax = plt.subplots(figsize=(max(8, n_coeffs * 0.3), 5)) # Adjust width

        # Stem plot for the values
        markerline, stemlines, baseline = ax.stem(l_indices, values_to_plot, linefmt='grey', markerfmt='o', basefmt='black')
        plt.setp(markerline, markersize=4, label=y_label) # Add label here for legend if needed
        plt.setp(stemlines, linewidth=1)
        plt.setp(baseline, linewidth=1)

        # Add error bars if they are valid
        plot_errors_valid = False
        if plot_errors_valid:
             # Only plot error bars where the calculated error is positive
             non_zero_error_mask = error_values > 1e-12
             if np.any(non_zero_error_mask):
                  ax.errorbar(l_indices[non_zero_error_mask], values_to_plot[non_zero_error_mask],
                              yerr=error_values[non_zero_error_mask],
                              fmt='none', ecolor='red', elinewidth=1, capsize=3, label='Std. Error')

        # --- Labels, Ticks, Title ---
        ax.set_xlabel("Zernike Index (l)")
        ax.set_ylabel(y_label)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Ticks and labels showing l and (n,m)
        from matplotlib.ticker import MaxNLocator # Ensure import is available
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        xtick_labels = [f"{l}\n({n},{m})" for l, (n, m) in enumerate(indices)]
        ax.set_xticks(l_indices)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)

        if plot_type == 'sq' or plot_type == 'abs':
            ax.set_ylim(bottom=0) # Ensure y-axis starts at 0 for non-negative values

        # Generate default title if none provided
        if title is None:
            title = f"Zernike Spectrum ({plot_type.capitalize()} Value) - Fit (n_max={self.n_max})"
        ax.set_title(title)

        # Add legend only if error bars were actually plotted
        if plot_errors_valid and np.any(non_zero_error_mask):
             ax.legend(fontsize='small')

        plt.tight_layout()

        # --- Save or Show ---
        if filename:
            try:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {filename}")
            except Exception as e:
                print(f"ERROR saving plot to {filename}: {e}")
            finally:
                plt.close(fig) # Close the figure after attempting to save
        else:
            plt.show()

# --- Example Usage ---

def simulate_gaussian_data(n_points=5000, x_range=(-5, 5), y_range=(-5, 5),
                           true_xc=0.8, true_yc=-0.5, sigma_x=1.5, sigma_y=2.0,
                           amplitude=1.0, noise_level=0.05, seed=None):
    """Generates sample 3D data resembling a noisy Gaussian blob."""
    if seed is not None:
        np.random.seed(seed) # For reproducibility
    x_orig = np.random.uniform(x_range[0], x_range[1], n_points)
    y_orig = np.random.uniform(y_range[0], y_range[1], n_points)
    z_true = amplitude * np.exp(
        -(((x_orig - true_xc)**2 / (2 * sigma_x**2)) +
          ((y_orig - true_yc)**2 / (2 * sigma_y**2)))
    )
    z_data = z_true + np.random.normal(0, noise_level, n_points)
    print(f"Generated {n_points} simulated data points.")
    print(f"True Centroid: ({true_xc:.2f}, {true_yc:.2f})")
    return x_orig, y_orig, z_data

if __name__ == "__main__":

    # --- Configuration ---
    SIMULATE_DATA = True # <<< SET TO FALSE TO LOAD YOUR OWN DATA
    N_MAX_FIT = 4       # Max Zernike radial order
    R_MAX_FIT = 5.0     # Normalization radius for fitting
                        # Should encompass the main features of your data
                        # relative to its expected center.

    # --- Get Data ---
    if SIMULATE_DATA:
        X, Y, Z = simulate_gaussian_data(
            n_points=5000,
            x_range=[-5, 5], y_range=[-5, 5],
            true_xc=0.8, true_yc=-0.5,
            sigma_x=1.5, sigma_y=2.0,
            amplitude=2.0, # Changed amplitude for better visualization
            noise_level=0.1,
            seed=42
        )
    else:
        # --- !!! LOAD YOUR REAL DATA HERE !!! ---
        # Replace this with your actual data loading logic
        # Ensure X, Y, Z are 1D numpy arrays of the same length
        print("Loading real data...")
        # Example placeholder:
        try:
            # data = np.loadtxt("your_data.txt", delimiter=',', skiprows=1)
            # X = data[:, 0]
            # Y = data[:, 1]
            # Z = data[:, 2]
            # print(f"Loaded {len(X)} points from your_data.txt")

            # If using data from variables already in scope (like from a previous cell)
            # Make sure they are named X, Y, Z or assign them:
            # X = my_x_data_variable
            # Y = my_y_data_variable
            # Z = my_z_data_variable
            raise NotImplementedError("Please replace placeholder with your data loading code.")

        except FileNotFoundError:
             print("Error: Data file not found. Please check the path.")
             exit()
        except Exception as e:
             print(f"Error loading data: {e}")
             exit()
        # --- END OF DATA LOADING ---


    # --- Create and Run Fitter ---
    fitter = ZernikeFitter(n_max=N_MAX_FIT, r_max=R_MAX_FIT)

    # Optional: Define custom initial guesses or bounds if defaults are poor
    # initial_guess = {'xc': 0.5, 'yc': -0.2, 'coeffs': [0]*fitter.n_coeffs} # Example
    # bounds = {'lower': [-1, -1, -10,...], 'upper': [1, 1, 10,...]} # Example
    # fit_success = fitter.fit(X, Y, Z, initial_guess=initial_guess, bounds=bounds)

    fit_success = fitter.fit(X, Y, Z, verbose=True) # Use default guesses/bounds

    # --- Display Results ---
    if fit_success:
        results = fitter.get_results(include_coeffs_table=True)

        print("\n--- Fit Results Summary ---")
        print(f"Success: {results['success']}")
        print(f"Message: {results['message']}")
        print(f"Fitted Centroid Xc = {results['fitted_xc']:.4f} +/- {results['err_xc']:.4f}")
        print(f"Fitted Centroid Yc = {results['fitted_yc']:.4f} +/- {results['err_yc']:.4f}")
        print(f"RMSE: {results['rmse']:.4f}")
        print(f"R-squared: {results['r_squared']:.4f}")
        print(results.get("coefficients_table", "Coefficient table not generated.")) # Safely get table

        # --- Optional: Predict on a new grid (Example) ---
        # x_new = np.linspace(-R_MAX_FIT, R_MAX_FIT, 20)
        # y_new = np.zeros_like(x_new) # Predict along x-axis
        # z_predicted_new = fitter.predict(x_new, y_new)
        # if z_predicted_new is not None:
        #     print("\n--- Prediction Example (Z values along x-axis at y=0) ---")
        #     for xi, zi in zip(x_new, z_predicted_new):
        #          print(f"  X={xi:+.2f}, Predicted Z={zi:+.4f}")

        # --- Optional: Plot Comparison ---
        try:
             fitter.plot_fit_comparison(X, Y, Z, n_grid=50) #, filename="zernike_fit_comparison.png")
        except Exception as e:
             print(f"\nError during plotting: {e}")
             traceback.print_exc()

    else:
        print("\n--- Fit Failed ---")
        print(f"Message: {fitter.message}")
        
# Make ZERNPY_AVAILABLE accessible after class definition if needed elsewhere
ZERNPY_AVAILABLE = ZERNPY_AVAILABLE