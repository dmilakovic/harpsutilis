#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 15:29:58 2025

@author: dmilakov
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import zernike # Install using: pip install zernike

# --- Parameters for Example Data ---
N_POINTS = 5000 # Number of data points
X_RANGE = [-5, 5]
Y_RANGE = [-5, 5]
TRUE_XC = 0.8   # True centroid X
TRUE_YC = -0.5  # True centroid Y
GAUSS_SIGMA_X = 1.5
GAUSS_SIGMA_Y = 2.0
GAUSS_AMPLITUDE = 1.0
NOISE_LEVEL = 0.05

# --- Generate Example Data ---
# Random points in the original X, Y space
X_orig = np.random.uniform(X_RANGE[0], X_RANGE[1], N_POINTS)
Y_orig = np.random.uniform(Y_RANGE[0], Y_RANGE[1], N_POINTS)

# Calculate Z using an offset Gaussian + noise
Z_true = GAUSS_AMPLITUDE * np.exp(
    -(((X_orig - TRUE_XC)**2 / (2 * GAUSS_SIGMA_X**2)) +
      ((Y_orig - TRUE_YC)**2 / (2 * GAUSS_SIGMA_Y**2)))
)
Z_data = Z_true + np.random.normal(0, NOISE_LEVEL, N_POINTS)

print(f"Generated {N_POINTS} data points.")
print(f"True Centroid: ({TRUE_XC:.2f}, {TRUE_YC:.2f})")

# Optional: Visualize the generated data
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_orig, Y_orig, Z_data, c=Z_data, cmap='viridis', s=5, alpha=0.6)
ax.set_xlabel('X original')
ax.set_ylabel('Y original')
ax.set_zlabel('Z data')
ax.set_title('Generated Sample Data')
plt.colorbar(sc, label='Z value')
plt.show()
#%%
def generate_zernike_indices(n_max):
    """Generates a list of (n, m) Zernike indices up to radial order n_max."""
    indices = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            indices.append((n, m))
    return indices

def calculate_zernike_basis(rho, phi, zernike_indices):
    """
    Calculates a basis matrix where each column is a Zernike polynomial
    evaluated at the given rho, phi coordinates.

    Args:
        rho (np.ndarray): Radial coordinates (normalized).
        phi (np.ndarray): Azimuthal coordinates (radians).
        zernike_indices (list): List of (n, m) tuples for Zernike modes.

    Returns:
        np.ndarray: Basis matrix (n_points, n_zernikes).
                    Returns None if rho/phi are invalid.
    """
    if rho is None or phi is None or len(rho) == 0 or len(phi) == 0:
        return None
    if len(rho) != len(phi):
         raise ValueError("rho and phi must have the same length")

    n_points = len(rho)
    n_zernikes = len(zernike_indices)
    basis_matrix = np.zeros((n_points, n_zernikes))

    cart = zernike.ZernikeCartesian() # Use the zernike library

    # Pre-calculate polynomials for unique (n,m) pairs efficiently if needed,
    # but direct calculation is often fine with the `zernike` library.

    # The `zernike` library expects X, Y normalized coordinates directly
    # Ensure rho <= 1 for valid calculation
    mask = rho <= 1.0
    x_norm = np.zeros_like(rho)
    y_norm = np.zeros_like(rho)
    x_norm[mask] = rho[mask] * np.cos(phi[mask])
    y_norm[mask] = rho[mask] * np.sin(phi[mask])

    # Use eval_cartesian which takes normalized x,y and the (n,m) list
    # This is generally more efficient than calculating one by one
    try:
        # Ensure indices is a list of tuples for eval_cartesian
        basis_matrix[mask, :] = cart.eval_cartesian(x_norm[mask], y_norm[mask], zernike_indices)
    except Exception as e:
        print(f"Error during Zernike evaluation: {e}")
        # Handle potential issues within the zernike library if necessary
        # Fallback to individual calculation if needed:
        # for i, (n, m) in enumerate(zernike_indices):
        #     z = zernike.Rnm(n, m)
        #     basis_matrix[mask, i] = z(rho[mask], phi[mask])
        pass # Or re-raise

    return basis_matrix
#%%
def zernike_model_with_centroid(xy_data, x_c, y_c, *coeffs):
    """
    Model function for curve_fit combining centroid shift, scaling,
    and Zernike polynomial summation.

    Args:
        xy_data (np.ndarray): Stacked array of original X and Y coordinates (2, n_points).
        x_c (float): X centroid offset.
        y_c (float): Y centroid offset.
        *coeffs (tuple): Zernike coefficients corresponding to zernike_indices_fit.

    Returns:
        np.ndarray: Predicted Z values (flattened).
    """
    X_orig = xy_data[0]
    Y_orig = xy_data[1]
    coeffs = np.array(coeffs) # Ensure coeffs is a numpy array

    if len(coeffs) != len(zernike_indices_fit):
        raise ValueError(f"Number of coefficients ({len(coeffs)}) does not match "
                         f"number of Zernike indices ({len(zernike_indices_fit)})")

    # --- Coordinate Transformation ---
    # 1. Apply centroid shift
    X_shifted = X_orig - x_c
    Y_shifted = Y_orig - y_c

    # 2. Scale to unit disk
    # R_max defines the radius in the *original* coordinate system
    # that maps to rho=1 in the normalized system.
    # Since X/Y go from -5 to 5, R_max should be 5.0.
    X_norm = X_shifted / R_MAX_FIT
    Y_norm = Y_shifted / R_MAX_FIT

    # 3. Convert to polar coordinates (rho, phi)
    rho = np.sqrt(X_norm**2 + Y_norm**2)
    phi = np.arctan2(Y_norm, X_norm)

    # --- Calculate Zernike Basis ---
    # Only calculate for points within the unit disk (rho <= 1)
    valid_mask = rho <= 1.0
    rho_valid = rho[valid_mask]
    phi_valid = phi[valid_mask]

    # Important: Ensure zernike_indices_fit is accessible (defined globally or passed)
    basis_matrix_valid = calculate_zernike_basis(rho_valid, phi_valid, zernike_indices_fit)

    # --- Predict Z ---
    Z_pred = np.zeros_like(X_orig)
    if basis_matrix_valid is not None and basis_matrix_valid.shape[0] > 0:
        Z_pred[valid_mask] = basis_matrix_valid @ coeffs

    # curve_fit expects a 1D array
    return Z_pred.ravel()
#%%
# --- Fitting Parameters ---
N_MAX_FIT = 4      # Maximum radial order for Zernike fit
R_MAX_FIT = 5.0    # Radius in original coordinates mapping to rho=1

# Generate the Zernike indices that will be fitted
zernike_indices_fit = generate_zernike_indices(N_MAX_FIT)
n_coeffs = len(zernike_indices_fit)

print(f"Fitting up to n_max = {N_MAX_FIT}")
print(f"Number of Zernike coefficients: {n_coeffs}")
print("Zernike (n, m) indices being fitted:")
print(zernike_indices_fit)

# --- Prepare Data for curve_fit ---
# curve_fit expects xdata as an array where columns are independent variables
# Here, we have X and Y, so we stack them.
xy_data_fit = np.vstack((X_orig.ravel(), Y_orig.ravel()))
z_data_fit = Z_data.ravel()

# --- Initial Guesses (p0) ---
# Centroid: Start near the center (0, 0) or use data mean if expected to be close
initial_x_c = 0.0 # np.average(X_orig, weights=Z_data) # Optional: weighted guess
initial_y_c = 0.0 # np.average(Y_orig, weights=Z_data) # Optional: weighted guess
# Zernike coeffs: Start with zeros, maybe guess piston (0,0) as mean Z
initial_coeffs = np.zeros(n_coeffs)
# Find index of (0,0) piston term if present
try:
    piston_index = zernike_indices_fit.index((0, 0))
    initial_coeffs[piston_index] = np.mean(z_data_fit)
except ValueError:
    piston_index = -1 # Piston term not included

p0 = [initial_x_c, initial_y_c] + list(initial_coeffs)

print(f"\nInitial guess (p0):")
print(f"  Xc={p0[0]:.3f}, Yc={p0[1]:.3f}")
print(f"  Coeffs=[{', '.join(f'{c:.3f}' for c in p0[2:])}]")


# --- Bounds (Optional but Recommended) ---
# Prevent centroid from going too wild, Zernikes usually unbounded
bounds_lower = [-R_MAX_FIT/2, -R_MAX_FIT/2] + [-np.inf] * n_coeffs
bounds_upper = [R_MAX_FIT/2, R_MAX_FIT/2] + [np.inf] * n_coeffs
bounds = (bounds_lower, bounds_upper)


# --- Run curve_fit ---
print("\nStarting curve_fit...")
try:
    # Increase maxfev if convergence is an issue
    popt, pcov = curve_fit(
        zernike_model_with_centroid,
        xy_data_fit,
        z_data_fit,
        p0=p0,
        bounds=bounds, # Comment out if you don't want bounds
        maxfev=10000 # Increase if needed
    )
    print("curve_fit completed successfully.")

    # --- Extract Results ---
    fitted_x_c = popt[0]
    fitted_y_c = popt[1]
    fitted_coeffs = popt[2:]

    # Calculate uncertainties (square root of diagonal of covariance matrix)
    perr = np.sqrt(np.diag(pcov))
    err_x_c = perr[0]
    err_y_c = perr[1]
    err_coeffs = perr[2:]

    print("\n--- Fit Results ---")
    print(f"Fitted Centroid Xc = {fitted_x_c:.4f} +/- {err_x_c:.4f}")
    print(f"Fitted Centroid Yc = {fitted_y_c:.4f} +/- {err_y_c:.4f}")
    print("\nFitted Zernike Coefficients:")
    for i, (n, m) in enumerate(zernike_indices_fit):
        print(f"  Z(n={n}, m={m}): {fitted_coeffs[i]:.4f} +/- {err_coeffs[i]:.4f}")

    # --- Evaluate Fit Quality (Optional) ---
    Z_pred_final = zernike_model_with_centroid(xy_data_fit, fitted_x_c, fitted_y_c, *fitted_coeffs)
    residuals = z_data_fit - Z_pred_final
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((z_data_fit - np.mean(z_data_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.4f}")


    # --- Visualize Fit (Optional) ---
    fig_fit = plt.figure(figsize=(10, 7))
    ax_fit = fig_fit.add_subplot(111, projection='3d')
    # Plot original data points
    ax_fit.scatter(X_orig, Y_orig, Z_data, c='blue', s=5, alpha=0.2, label='Original Data')

    # Create a grid for plotting the fitted surface
    n_grid = 50
    x_grid_orig = np.linspace(X_RANGE[0], X_RANGE[1], n_grid)
    y_grid_orig = np.linspace(Y_RANGE[0], Y_RANGE[1], n_grid)
    X_grid_orig, Y_grid_orig = np.meshgrid(x_grid_orig, y_grid_orig)
    xy_grid_fit = np.vstack((X_grid_orig.ravel(), Y_grid_orig.ravel()))

    Z_grid_pred = zernike_model_with_centroid(xy_grid_fit, fitted_x_c, fitted_y_c, *fitted_coeffs)
    Z_grid_pred = Z_grid_pred.reshape(X_grid_orig.shape) # Reshape back to 2D grid

    # Plot fitted surface
    surf = ax_fit.plot_surface(X_grid_orig, Y_grid_orig, Z_grid_pred, cmap='viridis', alpha=0.7, label='Fitted Surface')

    ax_fit.set_xlabel('X original')
    ax_fit.set_ylabel('Y original')
    ax_fit.set_zlabel('Z')
    ax_fit.set_title(f'Original Data and Zernike Fit (n_max={N_MAX_FIT}, R_max={R_MAX_FIT})')
    # Add a color bar which maps values to colors.
    # fig_fit.colorbar(surf, shrink=0.5, aspect=5) # surf doesn't have direct colorbar, use scatter like before
    # Add legend manually if needed, as surface doesn't have a label property directly usable in legend
    # ax_fit.legend() # Simple legend might only show scatter plot

    plt.show()


except RuntimeError as e:
    print(f"\nError during curve_fit: {e}")
    print("The fit failed to converge. Try:")
    print("  - Adjusting initial guesses (p0).")
    print("  - Changing the number of Zernike terms (N_MAX_FIT).")
    print("  - Increasing 'maxfev' in curve_fit.")
    print("  - Checking the data for NaNs or outliers.")
    print("  - Adjusting bounds if used.")
except Exception as e:
     print(f"\nAn unexpected error occurred: {e}")
     