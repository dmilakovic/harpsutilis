#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:22:25 2025

@author: dmilakov
"""

import jax.numpy as jnp

from fitsio import FITS
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import itertools, lmfit

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float, filters
from skimage.morphology import disk


#%%
lfc_filename = '/Users/dmilakov/projects/lfc/data/harps/raw/4bruce/HARPS.2015-04-17T00:00:41.445_lfc.fits'
bias_filename = '/Users/dmilakov/projects/lfc/data/harps/raw/4bruce/HARPS.2015-04-16T17:40:29.876_bias.fits'
with FITS(bias_filename,'r') as hdul:
    bias_blue = hdul[1].read()
    bias_red  = hdul[2].read()
    
with FITS(lfc_filename,'r') as hdul:
    data_blue = hdul[1].read()
    data_red  = hdul[2].read()
    
    unbiased_blue = data_blue - bias_blue
    unbiased_red  = data_red  - bias_red
    
#%%
from skimage.morphology import disk
neighborhood = disk(radius=20)  # "selem" is often the name used for "structuring element"
# maxfilter = filters.rank.mean_bilateral(data_red, neighborhood)
# plt.imshow(maxfilter, aspect='equal')


#%%
qp15 = np.quantile(data_red,0.5)
cut = np.where((data_red <= 320))
plt.figure()
data_red[cut]=0.
coordinates = peak_local_max(data_red, min_distance=5)
# plt.imshow(np.log10(data_red))
plt.imshow(data_red)
plt.scatter(coordinates[:,1],coordinates[:,0],ec='r',fc='None',marker='s')   

sorter=np.argsort(coordinates,axis=0)
sorted_coords=coordinates[sorter] 
# for i,(x0, y0) in enumerate(sorted_coords):
#     plt.text(x0,y0,f'{i+1}',)

#%%
stamps = []
dist = 5
for y0, x0 in coordinates:
    
    lef = x0 - dist
    rig = x0 + dist +1
    top = y0 + dist +1
    bot = y0 - dist
    
    stamps.append([(bot,top),(lef,rig)])

#%%
from lmfit.lineshapes import lorentzian

def gaussian2d(x,y,amplitude,centerx,centery,sigmax,sigmay,theta=0,zero_offset=0):
    '''
    Modified from https://stackoverflow.com/questions/30420081/rotating-a-gaussian-function-matlab

    Parameters
    ----------
    x : array, x-coordinates.
    y : array, y-coordinates.
    amplitude : float
    mean : length 2 tuple, mean coordinates (x0,y0)
    sigma : length 2 tuple, standard deviations (sigma_x,sigma_y)
    theta : angle between the major axis and positive x direction, in degrees.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    assert (theta<=45 and theta>=-45), "Angle must be +- 45 degrees"
    # assert len(mean)==2, "Unexpected length for mean"
    # assert len(sigma)==2, "Unexpected length for sigma"
    
    # unpack mean coordinates and standard deviation values
    # x0, y0 = mean
    # sigma_x, sigma_y = sigma
    # convert theta from degrees to radians
    zo = zero_offset
    theta = - theta / 180 * jnp.pi 
    
    center_x = centerx
    center_y = centery
    sigma_x = sigmax
    sigma_y = sigmay
    
    normA = amplitude / (2*jnp.pi * sigma_x * sigma_y)
    
    a = 0.5  * ( (jnp.cos(theta)/sigma_x)**2 + (jnp.sin(theta)/sigma_y)**2)
    b = 0.25 * ( - (jnp.sin(2*theta)/sigma_x**2) + (jnp.sin(2*theta)/sigma_y**2))
    #b = 0
    c = 0.5  * ( (jnp.sin(theta)/sigma_x)**2 + (jnp.cos(theta)/sigma_y)**2)
    
    return zo + normA*jnp.exp(-0.5*(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def lorentzian2d(x, y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1.,
                 theta=0):
    """Return a two dimensional lorentzian.

    The maximum of the peak occurs at ``centerx`` and ``centery``
    with widths ``sigmax`` and ``sigmay`` in the x and y directions
    respectively. The peak can be rotated by choosing the value of ``rotation``
    in radians.
    """
    xp = (x - centerx)*np.cos(theta/180*np.pi) - (y - centery)*np.sin(theta/180*np.pi)
    yp = (x - centerx)*np.sin(theta/180*np.pi) + (y - centery)*np.cos(theta/180*np.pi)
    R = (xp/sigmax)**2 + (yp/sigmay)**2

    return 2*amplitude*lorentzian(R)/(np.pi*sigmax*sigmay)

def confidence_ellipse(ax, x0, y0, sigma_x, sigma_y, theta, cl = 0.68, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    import matplotlib.transforms as transforms
    from matplotlib.patches import Ellipse
    from scipy.stats import chi2
    
    rho = np.tan(theta)
    sigma = np.matrix([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])
    eig=np.linalg.eig(sigma)
    eigenval = eig.eigenvalues
    eigenvec = eig.eigenvectors
    
    chi2_dist = chi2(2)
    patches = []
    for cl_ in np.atleast_1d(cl):
        sf        = chi2_dist.ppf(cl_)
        
        l         = np.sqrt(np.abs(eig.eigenvalues) * sf) 
        
        # ell_radius_x = np.sqrt(2 * sigma_x**2 * jnp.log(1./2*jnp.pi*n_std*sigma_x*sigma_y))
        # ell_radius_y = np.sqrt(2 * sigma_y**2 * jnp.log(1./2*jnp.pi*n_std*sigma_x*sigma_y))
        ell_radius_x = l[0]
        ell_radius_y = l[1]
        ellipse = Ellipse((x0, y0), width=ell_radius_x , height=ell_radius_y ,
                          angle=theta,
                          facecolor=facecolor, **kwargs)
        
        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = sigma_x #* n_std
        # mean_x = np.mean(x)
    
        # # calculating the standard deviation of y ...
        scale_y = sigma_x #* n_std
        # mean_y = np.mean(y)
    
        # transf = transforms.Affine2D() \
        #     .rotate_deg(45) \
        #     .scale(scale_x, scale_y) \
        #     .translate(x0, y0)
    
        # ellipse.set_transform(transf + ax.transData)
        print(ellipse)
        patches.append(ax.add_patch(ellipse))
    return patches

x0 = 0
y0 = 0
sigma_x = 0.8
sigma_y = 0.5
theta = 40
xv,xu = np.meshgrid(np.linspace(-6,6,100),np.linspace(-6,6,100))
g2d=gaussian2d(xv,xu,amplitude=1,
               centerx=x0,
               centery=y0,
               sigmax=sigma_x,
               sigmay=sigma_y,
               theta=theta,zero_offset=0)
fig = plt.figure()
ax = fig.subplots(1)
ax.imshow(g2d,origin='lower',
            extent=(-2,2,-2,2))
confidence_ellipse(ax, 
                    x0=x0,y0=y0,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    theta=theta,
                    cl=[0.68,0.95,0.99],
                    edgecolor='red')
ax.set_title(f'Theta = {theta} deg')

#%%    
    
    
    
def barycentre(mass,position,plot=False,verbose=False):
    xsum = 0
    ysum = 0
    M = np.sum(mass)
    
    xpos = np.unique(position[:,1])
    ypos = np.unique(position[:,0])
    # print(xpos,ypos)
    
    for (y,x), m in zip(positions,np.ravel(mass)):
        if verbose:
            print(f'{x:4d}  {y:4d}  {m:12.8f}')
        xsum += x*m
        ysum += y*m
    
    if plot:
        plt.figure()
        plt.imshow(mass,
                   extent=(np.min(xpos),np.max(xpos),
                           np.min(ypos),np.max(ypos)),
                   origin='lower')
        plt.axhline(ysum/M,c='r')
        plt.axvline(xsum/M,c='r')
        plt.show()
    return xsum/M, ysum/M

centred_data = []

color = 'magenta'

plot_result=True
fit_model = True

count = 0
for stamp in stamps:        
    (bot,top),(lef,rig) = stamp
    if bot>=1455 and top<=1830 and lef>=912 and rig<=930:
        count +=1
        
        data_stamp = data_red[slice(bot,top),slice(lef,rig)] + 1e-6
        sumflx = np.sum(data_stamp)
        normflx = data_stamp/sumflx
        # normflx = data_stamp / np.max(data_stamp)
        
        xpos=np.arange(lef,rig)
        ypos=np.arange(bot,top)
        positions=np.array([*itertools.product(ypos,xpos)])
        bary_x, bary_y = barycentre(data_stamp,positions,plot=False)
        
        # data_ = data_stamp
        data_ = normflx
        if fit_model:
            
            model = lmfit.Model(gaussian2d,independent_vars=['x','y'])
            # model = lmfit.models.Gaussian2dModel()
            # model = lmfit.Model(lorentzian2d, independent_vars=['x', 'y'])
            params = model.make_params(centerx=dict(value=0),
                                       centery=dict(value=0),
                                       amplitude=dict(value=10*np.max(data_),min=0),
                                       theta=dict(value=0.,vary=False, min=-45, max=45),
                                       sigmax=dict(value=1.2,min=0.),
                                       sigmay=dict(value=1.2,min=0.),
                                       # zero_offset=dict(value=320,min=320,max=500)
                                       )
            # params['amplitude'].set(value=np.max(data_stamp),min=0)
            # params['theta'].set(value=0.0,min=0,max=90)
            # params['sigma_x'].set(value=1.,min=0)
            # params['sigma_y'].set(value=1.,min=0)
            
            #result = model.fit(normflx, x=xpos,y=ypos, params=params, weights=1/np.sqrt(normflx))
            weights=np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0.5,0,0,0,0,0,0],
                              [0,0,0,0,0,  1,1,1,0,0,0,0,0],
                              [0,0,0,0,0.5,1,1,1,0.5,0,0,0,0],
                              [0,0,0,0,0,  1,1,1,0,0,0,0,0],
                              [0,0,0,0,0,0,0.5,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0,0,0,0]])
            result = model.fit(data_, 
                              # method='emcee',
                               x=xpos-bary_x,y=ypos-bary_y, 
                               params=params,
                               weights = np.sqrt(data_),
                               )
            
            lmfit.report_fit(result)
        # width, height = rig-lef, top-bot
        
        # lbl = ndi.label(normflx)[0]
        # bary_x, bary_y = ndi.center_of_mass(normflx)
        
        
        centred_data_ = np.dstack([positions[:,0]-bary_y,
                                   positions[:,1]-bary_x,
                                   np.ravel(data_)]
                                  )
        centred_data.append(centred_data_)
        
        print(bary_x,bary_y)
        if plot_result:
            extent=(lef-bary_x,rig-bary_x-1,
                    bot-bary_y,top-bary_y-1)
            print(extent)
            
            
            
            
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # ax.imshow(data_,
            #             origin='lower',
            #             # extent = (lef,rig,bot,top)
            #             # extent=(-bary_x,width-bary_x,
            #             #         -bary_y,height-bary_y),
            #             extent=extent
            #             )
            ax.scatter(positions[:,1]-bary_x,
                       positions[:,0]-bary_y,
                       data_,
                       ls='-')
            
            # ax.axvline(0,c='r')
            # ax.axhline(0,c='r')
            ax.set_xlim(-6,6)
            ax.set_ylim(-6,6)
            
            if fit_model:
                X_ = np.linspace(-6,6,101)
                Y_ = np.linspace(-6,6,101)
                X, Y = np.meshgrid(X_,Y_)
                values_dict = result.best_values
                amplitude   = values_dict['amplitude']
                x0          = values_dict['centerx']
                y0          = values_dict['centery']
                sigma_x     = values_dict['sigmax']
                sigma_y     = values_dict['sigmay']
                # theta       = values_dict['theta']
                # zo          = values_dict['zero_offset']
                fit = model.func(X, Y, 
                                 amplitude = amplitude,
                                 centerx = x0 ,
                                 centery = y0 ,
                                 sigmax = sigma_x,
                                 sigmay = sigma_y,
                                 # theta = theta,
                                 # zero_offset= zo
                                 )
                cmap = 'coolwarm'
                ax.plot_surface(X,Y,fit,color='C1',alpha=0.3)
                # ax.contourf(X,Y,fit,zdir = 'z', offset = -0.005, cmap=cmap)
                # ax.contourf(X,Y,fit,zdir = 'x', offset = -6, cmap=cmap)
                # ax.contourf(X,Y,fit,zdir = 'y', offset = 6, cmap=cmap)
                
                kx = ky = 6
                soln=polyfit2d(xpos-bary_x, ypos-bary_y, data_, kx=kx, ky=ky)
                fitted_surf = np.polynomial.polynomial.polygrid2d(X_, Y_, soln[0].reshape((kx+1,ky+1)))
                ax.plot_surface(X,Y,fitted_surf,color='C2',alpha=0.3)
                ax.contourf(X,Y,fitted_surf,zdir = 'z', offset = -0.005, cmap=cmap)
                ax.contourf(X,Y,fitted_surf,zdir = 'x', offset = -6, cmap=cmap)
                ax.contourf(X,Y,fitted_surf,zdir = 'y', offset = 6, cmap=cmap)
                
                # ax.contour(X,Y,fit)
                
                # confidence_ellipse(ax, 
                #                     x0=x0,y0=y0,
                #                     sigma_x=sigma_x,
                #                     sigma_y=sigma_y,
                #                     theta=theta,
                #                     cl=[0.68,0.95,0.99],
                #                     edgecolor=color)
                
                
                # plt.axvline(x0,c=color)
                # plt.axhline(y0,c=color)
                
        if count>=3:
            break
    else:
        continue

centred_data = np.vstack(centred_data)

#%%
from scipy.special import factorial

# --------------------------
# Zernike Polynomial Functions
# --------------------------

def zernike_radial(n, m, rho):
    """
    Compute the radial component R_n^m(rho) of the Zernike polynomial.

    Parameters:
        n (int): radial order (non-negative integer)
        m (int): azimuthal frequency (|m| <= n and n - |m| even)
        rho (ndarray): radial coordinate (0 <= rho <= 1)

    Returns:
        ndarray: values of the radial polynomial at the given rho.
    """
    m = abs(m)
    R = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s *
             factorial(n - s) /
             (factorial(s) *
              factorial((n + m) // 2 - s) *
              factorial((n - m) // 2 - s)))
        R += c * np.power(rho, n - 2 * s)
    return R

def zernike(n, m, rho, theta):
    """
    Compute the full Zernike polynomial Z_n^m(rho, theta).

    For m >= 0, the angular part is cos(m * theta); for m < 0, it is sin(|m| * theta).

    Parameters:
        n (int): radial order.
        m (int): azimuthal order.
        rho (ndarray): radial coordinate (normalized to 1).
        theta (ndarray): angular coordinate.

    Returns:
        ndarray: Zernike polynomial evaluated at (rho, theta).
    """
    if m >= 0:
        return zernike_radial(n, m, rho) * np.cos(m * theta)
    else:
        return zernike_radial(n, -m, rho) * np.sin(-m * theta)

def generate_zernike_basis(image_shape, max_order):
    """
    Generate a set of Zernike basis functions up to the given maximum order
    on a square image grid. The basis functions are computed only on the unit disk.

    Parameters:
        image_shape (tuple): shape of the image (height, width).
        max_order (int): maximum radial order n.

    Returns:
        tuple: (basis, indices, mask)
            basis: ndarray of shape (num_basis, height, width)
            indices: array of (n, m) tuples for each basis function.
            mask: boolean ndarray indicating points inside the unit disk.
    """
    # Create coordinate grid
    y, x = np.indices(image_shape)
    # Define the center as the middle of the image
    cy, cx = image_shape[0] // 2, image_shape[1] // 2

    # Normalize radius: we use half the image size as the unit radius.
    r = np.sqrt((x - cx)**2 + (y - cy)**2) / (image_shape[0] // 2)
    theta = np.arctan2(y - cy, x - cx)

    # Use only points inside the unit disk.
    mask = r <= 1

    basis = []
    indices = []
    # Loop over orders: for each n, valid m values are -n, -n+2, ..., n-2, n.
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            Z = np.zeros_like(r)
            Z[mask] = zernike(n, m, r[mask], theta[mask])
            basis.append(Z)
            indices.append((n, m))
    return np.array(basis), np.array(indices), mask

def fit_zernike(image, basis, mask):
    """
    Fit an image (assumed to be defined on a disk) by projecting it onto each
    Zernike basis function. The coefficients are computed via a least-squares projection.

    Parameters:
        image (ndarray): 2D image.
        basis (ndarray): array of basis functions (num_basis, height, width).
        mask (ndarray): boolean mask indicating valid points (inside unit disk).

    Returns:
        ndarray: array of Zernike coefficients.
    """
    coeffs = []
    for Z in basis:
        numerator = np.sum(image[mask] * Z[mask])
        denominator = np.sum(Z[mask] ** 2)
        coeffs.append(numerator / denominator if denominator != 0 else 0)
    return np.array(coeffs)

def reconstruct_image(coeffs, basis, mask):
    """
    Reconstruct the image using the Zernike coefficients and basis functions.

    Parameters:
        coeffs (ndarray): Zernike coefficients.
        basis (ndarray): array of basis functions.
        mask (ndarray): boolean mask indicating valid points.

    Returns:
        ndarray: reconstructed image.
    """
    rec = np.sum(coeffs[:, None, None] * basis, axis=0)
    rec[~mask] = 0
    return rec

# --------------------------
# Main Code to Process the PSF "peak"
# --------------------------

# Assume "peak" is your square PSF image as a NumPy array.
# If you don't already have one, you can uncomment the following lines to create a dummy 13x13 image.
# For example, a simple PSF with a bright central peak:
#
# peak = np.zeros((13, 13))
# peak[6, 6] = 1  # a delta-like function at the center
#
# Alternatively, you might have a more realistic PSF already loaded into "peak".

# Uncomment below to create a dummy example if needed:
# peak = np.zeros((13, 13))
# peak[6, 6] = 1

# Use the shape of 'peak'
peak = data_
image_shape = peak.shape

# Set the maximum Zernike order. For a 13x13 image, using up to n=6 is a reasonable starting point.
max_order = 8

# Generate the Zernike basis functions based on the shape of "peak"
basis, indices, mask = generate_zernike_basis(image_shape, max_order)

# Compute the Zernike coefficients by projecting the image onto the basis functions.
coeffs = fit_zernike(peak, basis, mask)

# Print the best fit coefficients along with their (n, m) indices.
print("Best Fit Zernike Coefficients:")
for (n, m), c in zip(indices, coeffs):
    print(f"n = {n:2d}, m = {m:2d}: {c:.6f}")

# Reconstruct the PSF image from the Zernike coefficients.
reconstruction = reconstruct_image(coeffs, basis, mask)

# Display the original PSF and the reconstructed PSF side by side.
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(peak, cmap='hot')
axes[0].set_title('Original PSF (peak)')
axes[1].imshow(reconstruction, cmap='hot')
axes[1].set_title('Reconstructed PSF')
plt.tight_layout()
plt.show()


#%%

import scipy.optimize as opt
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()


initial_guess = (1,0,0,1,1,0,0)
xv,yv=np.meshgrid(xpos-bary_x, ypos-bary_y)
popt, pcov = opt.curve_fit(twoD_Gaussian, (xv, yv), data_.ravel(), p0=initial_guess)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.imshow(data_,
#             origin='lower',
#             # extent = (lef,rig,bot,top)
#             # extent=(-bary_x,width-bary_x,
#             #         -bary_y,height-bary_y),
#             extent=extent
#             )
ax.scatter(positions[:,1]-bary_x,
           positions[:,0]-bary_y,
           data_,
           ls='-')

# ax.axvline(0,c='r')
# ax.axhline(0,c='r')
ax.set_xlim(-6,6)
ax.set_ylim(-6,6)

X_ = np.linspace(-6,6,101)
Y_ = np.linspace(-6,6,101)
X, Y = np.meshgrid(X_,Y_)
model_ = twoD_Gaussian((X,Y), *popt)

ax.plot_surface(X,Y,model_.reshape(101,101),color='C1',alpha=0.3)    
    
#%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(centred_data[...,0],
           centred_data[...,1],
           centred_data[...,2],
           marker='.')