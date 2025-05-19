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
qp15 = np.quantile(data_red,0.5)
cut = np.where((data_red <= 320))

data_red[cut]=0.
coordinates = peak_local_max(data_red, min_distance=5)
# plt.imshow(np.log10(data_red))


sorter=np.argsort(coordinates,axis=0)
sorted_coords=coordinates[sorter] 
# for i,(x0, y0) in enumerate(sorted_coords):
#     plt.text(x0,y0,f'{i+1}',)

def transform_coords(points_int):
    points_flt = points_int.astype(float)

    points_flt[:, 0] /= 2148.  # Normalize x
    points_flt[:, 1] /= 4096.  # Normalize y
    return points_flt

def inverse_transform_coords(points_flt):
    points_int = np.zeros_like(points_flt,dtype=int)
    
    points_int[:,0] = points_flt[:,0]*2148
    points_int[:,1] = points_flt[:,1]*4096

    return points_int    


plt.figure()
plt.imshow(data_red.T, cmap=plt.cm.jet)
plt.scatter(coordinates[:,0],coordinates[:,1],ec='r',fc='None',marker='s')   

#%% DO NOT USE
# test = True
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import cKDTree
# from sklearn.cluster import AgglomerativeClustering
# import networkx as nx

# def group_curved_lines(points, neighbor_radius=15, min_curve_size=20, n_clusters=None):
#     """
#     Groups (x, y) coordinates into curved lines using agglomerative clustering and graph merging.

#     Parameters:
#     - points: np.array of shape (n, 2), where each row is (x, y)
#     - neighbor_radius: Distance threshold for linking points into the same curve
#     - min_curve_size: Minimum number of points required to keep a curve
#     - n_clusters: Expected number of curves (if known, otherwise automatic)

#     Returns:
#     - line_groups: List of np.arrays, each containing points for a detected curve
#     """

#     # Step 1: Agglomerative Clustering to find initial clusters
#     clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(points)
#     labels = clustering.labels_

#     # Step 2: Build nearest-neighbor graphs for each cluster
#     clusters = {}
#     for i, label in enumerate(labels):
#         if label not in clusters:
#             clusters[label] = []
#         clusters[label].append(points[i])

#     line_groups = []
#     for cluster_points in clusters.values():
#         if len(cluster_points) < min_curve_size:
#             continue  # Remove small noise clusters

#         cluster_points = np.array(cluster_points)
#         tree = cKDTree(cluster_points)

#         # Build graph: Connect nearby points
#         G = nx.Graph()
#         for i, p in enumerate(cluster_points):
#             neighbors = tree.query_ball_point(p, neighbor_radius)
#             for j in neighbors:
#                 if i != j:
#                     G.add_edge(tuple(cluster_points[i]), tuple(cluster_points[j]))

#         # Extract connected components (each is a detected curve)
#         for component in nx.connected_components(G):
#             line_groups.append(np.array(list(component)))

#     return line_groups

# # Example Usage
# if test:
#     # Generate synthetic slightly curved lines
#     t = np.linspace(0, np.pi, 50)
#     x1, y1 = 50 + 30 * np.cos(t), 50 + 40 * t / np.pi
#     x2, y2 = 100 + 30 * np.cos(t), 60 + 40 * t / np.pi
#     x3, y3 = 150 + 30 * np.cos(t), 70 + 40 * t / np.pi

#     points = np.vstack((np.column_stack((x1, y1)), 
#                         np.column_stack((x2, y2)), 
#                         np.column_stack((x3, y3))))

#     curves = group_curved_lines(coordinates, n_clusters=171)  # Set number of expected curves

#     # Plot results
#     plt.figure(figsize=(8, 6))
#     plt.scatter(coordinates[:,0],coordinates[:,1],marker='.')
#     for i, curve in enumerate(curves):
#         plt.plot(curve[:, 0], curve[:, 1], marker='o', mfc=None,linestyle='-', label=f"Curve {i}")
    
#     plt.legend()
#     plt.gca().invert_yaxis()  # Match image coordinates
#     plt.title("Curved Line Clustering with Graph Merging")
#     plt.show()

#%% USE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def sort_and_cluster_lines(points, eps_x=0.008, eps_y=0.004, min_samples=5):
    """
    Groups (x, y) coordinates into horizontal curved lines using DBSCAN with independent x/y eps values.

    Parameters:
    - points: np.array of shape (n, 2), where each row is (x, y)
    - eps_x: Clustering threshold for x-direction
    - eps_y: Clustering threshold for y-direction
    - min_samples: Minimum points in a cluster for DBSCAN

    Returns:
    - line_groups: List of np.arrays, each containing points for a detected curve
    """

    # Step 1: Normalize x and y coordinates independently
    points_scaled = np.copy(points)
    points_scaled[:, 0] /= eps_x  # Scale x
    points_scaled[:, 1] /= eps_y  # Scale y

    # Step 2: Apply DBSCAN with uniform eps=1 (since we've scaled)
    clustering = DBSCAN(eps=1, min_samples=min_samples).fit(points_scaled)
    labels = clustering.labels_

    # Step 3: Extract clusters
    unique_labels = set(labels) - {-1}  # Ignore noise (-1)
    # line_groups = [points[labels == label] for label in unique_labels]
    line_groups = []
    for label in unique_labels:
        # Extract points belonging to the current cluster
        cluster_points = points[labels == label]
    
        # Sort the cluster points by x-coordinate
        cluster_points_sorted = cluster_points[np.argsort(cluster_points[:, 0])]
    
        # Store the sorted cluster
        line_groups.append(cluster_points_sorted)
        

    return line_groups

# Example Usage
test = True
if test:
    # Generate synthetic horizontal curved lines
    t = np.linspace(0, np.pi, 50)
    x1, y1 = 50 + 30 * np.cos(t), 50 + 40 * t / np.pi
    x2, y2 = 100 + 30 * np.cos(t), 60 + 40 * t / np.pi
    x3, y3 = 150 + 30 * np.cos(t), 70 + 40 * t / np.pi

    # Simulate real-world scaling
    points = np.vstack((np.column_stack((x1, y1)), 
                        np.column_stack((x2, y2)), 
                        np.column_stack((x3, y3))))
    points = transform_coords(coordinates)

    # Apply clustering with different eps values for x and y
    curves = sort_and_cluster_lines(points, eps_x=0.002, eps_y=0.02, min_samples=5)

    # Plot results
    plt.figure(figsize=(14, 12))
    plt.scatter(points[:, 0], points[:, 1], marker='.', alpha=0.3, label="Original Points")
    for i, curve in enumerate(curves):
        # sorter = np.argsort(curve[:,0])
        linestyle = '-' if i%2 == 0 else '--'
        plt.plot(curve[:, 0], curve[:, 1], marker='o', fillstyle='none', 
                 linestyle=linestyle, label=f"Curve {i}")

    # plt.legend()
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.title("Curved Line Detection with Independent X/Y Clustering")
    plt.tight_layout()
    plt.show()



#%%
from lmfit.lineshapes import lorentzian

def gaussian2d(xy,amplitude,centerx,centery,sigmax,sigmay,theta=0,zero_offset=0):
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
    x, y = xy
    # assert (theta<=45 and theta>=-45), "Angle must be +- 45 degrees"
    # assert len(mean)==2, "Unexpected length for mean"
    # assert len(sigma)==2, "Unexpected length for sigma"
    
    # unpack mean coordinates and standard deviation values
    # x0, y0 = mean
    # sigma_x, sigma_y = sigma
    # convert theta from degrees to radians
   
    
    center_x = float(centerx)
    center_y = float(centery)
    sigma_x = float(sigmax)
    sigma_y = float(sigmay)
    
    normA = amplitude / (2*jnp.pi * sigma_x * sigma_y)
    
    a = 0.5  * ( (jnp.cos(theta)/sigma_x)**2 + (jnp.sin(theta)/sigma_y)**2)
    b = 0.25 * ( - (jnp.sin(2*theta)/sigma_x**2) + (jnp.sin(2*theta)/sigma_y**2))
    c = 0.5  * ( (jnp.sin(theta)/sigma_x)**2 + (jnp.cos(theta)/sigma_y)**2)
    # a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    # c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    value = zo + amplitude*jnp.exp( - (a*((x-x0)**2) + \
                                       2*b*(x-x0)*(y-y0) + \
                                       c*((y-y0)**2)))
    
    
    # a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    # b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    # c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    # g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
    #                         + c*((y-yo)**2)))
    
    
    return value.ravel()

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
    
    rho = 0
    sigma = np.matrix([[sigma_x**2, rho*sigma_x*sigma_y],
                        [rho*sigma_x*sigma_y, sigma_y**2]])
    eig=np.linalg.eig(sigma)
    eigenval = eig.eigenvalues
    eigenvec = eig.eigenvectors
    
    chi2_dist = chi2(1)
    patches = []
    for cl_ in np.atleast_1d(cl):
        sf        = chi2_dist.ppf(cl_)
        l         = np.sqrt(np.abs(eig.eigenvalues) * sf) 
        
        # ell_radius_x = np.sqrt(2 * sigma_x**2 * jnp.log(1./2*jnp.pi*n_std*sigma_x*sigma_y))
        # ell_radius_y = np.sqrt(2 * sigma_y**2 * jnp.log(1./2*jnp.pi*n_std*sigma_x*sigma_y))
        ell_radius_x = l[0]
        ell_radius_y = l[1]
        # ell_radius_x = sigma_x * sf
        # ell_radius_y = sigma_y * sf
        ellipse = Ellipse((x0, y0), width=ell_radius_x , height=ell_radius_y ,
                          angle= - theta/np.pi * 180,
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
        # print(ellipse)
        patches.append(ax.add_patch(ellipse))
    return patches

x0 = 0
y0 = 0
sigma_x = 0.8
sigma_y = 0.5
zo = 0.
theta_deg = -40
theta_rad = theta_deg / 180 * np.pi
npts = 100
xv,xu = np.meshgrid(np.linspace(-6,6,npts),np.linspace(-6,6,npts))
g2d=gaussian2d((xv,xu),amplitude=1,
               centerx=x0,
               centery=y0,
               sigmax=sigma_x,
               sigmay=sigma_y,
               theta=theta_rad,
               zero_offset=0).reshape(npts,npts)
fig = plt.figure()
ax = fig.subplots(1)
ax.imshow(g2d,origin='lower',
            extent=(-2,2,-2,2))
confidence_ellipse(ax, 
                    x0=x0,y0=y0,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    theta=theta_rad,
                    cl=[0.68,0.95,0.99],
                    edgecolor='red')
ax.set_title(f'Theta = {theta_deg} deg')

#%%    
    
import scipy.optimize as opt
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m

    Parameters
    ----------
    xy : tuple of shape (2,n) with the (x,y) coordinates.
    amplitude : float, amplitude of the Gaussian.
    xo : float, mean in x-direction.
    yo : float, mean in y-direction.
    sigma_x : float, standard deviation in x-direction.
    sigma_y : float, standard deviation in y-direction. 
    theta : float, rotation angle in unit radian.
    offset : float, value added to all elements of output array.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()    
    
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

def rotate(p, origin=(0, 0), theta=0):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    m, n = np.shape(p)
    if n>1:
        origin = np.vstack([origin for _ in range(n)])
    
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p).T
    
    return np.squeeze((R @ (p.T-o.T) + o.T).T).T
    

centred_data = []

color = 'magenta'
projection = '2d'

plot_result=True
fit_model = True




curves = sort_and_cluster_lines(points, eps_x=0.02, eps_y=0.002, min_samples=5)

# option_1 = dict(BOTTOM=1155, TOP=1830, LEFT=912, RIGHT=930)
# option_2 = dict(BOTTOM=2770, TOP=4080, LEFT=2027, RIGHT=2060)
# option_3 = dict(BOTTOM=0, TOP=1300, LEFT=2040, RIGHT=2060)
# option_3 = dict(BOTTOM=0, TOP=1300, LEFT=60, RIGHT=100)

# range_dict = option_3
nseg = 0
dist = 5
k = 4
for j, curve in enumerate(curves[k:k+1]):        
    LEFT = nseg*512 #0
    RIGHT = (nseg+1)*512
    DIST = 5
    curve_pix = inverse_transform_coords(curve)
    count = 0
    for coord_x, coord_y in curve_pix:
        if coord_x>=LEFT and coord_x<=RIGHT:
            lef = coord_x - dist
            rig = coord_x + dist +1
            top = coord_y + dist +1
            bot = coord_y - dist
    
        
            count +=1
            
            data_stamp = data_red[slice(lef,rig),slice(bot,top)] + 1e-6
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
                
                keys = ["A","x0","y0","sigma_x","sigma_y","theta","offset"]
                initial_guess = (1,0,0,1,1,0,0)
                xv,yv=np.meshgrid(xpos-bary_x, ypos-bary_y)
                try:
                    popt, pcov, infodict, mesg, ier = opt.curve_fit(twoD_Gaussian, 
                                               (xv, yv), 
                                               data_.ravel(), 
                                               p0=initial_guess, 
                                               full_output=True)
                except:
                    continue
                perr = np.sqrt(np.diag(pcov))
                print("{0:=>60s}".format(''))
                print(f"{mesg}")
                print(f"{ier}")
                for key, val, err in zip(keys,popt,perr):
                    print(f'{key:10s}{val:>12.4e}{err:12.4e} ({err/val:12.3%})')
                if perr[0]/popt[0]>1:
                    continue
                
            # width, height = rig-lef, top-bot
            
            # lbl = ndi.label(normflx)[0]
            # bary_x, bary_y = ndi.center_of_mass(normflx)
            
            center = [bary_y, bary_x]
            center = [bary_y+popt[2], bary_x+popt[1]]
            
            x4rot = positions[:,0]-center[0]
            y4rot = positions[:,1]-center[1]
            x_after_rot, y_after_rot = rotate((x4rot, y4rot), theta=-popt[5])
            
            centred_data_ = np.dstack([x_after_rot,
                                       y_after_rot,
                                       np.ravel(data_)]
                                      )
            centred_data.append(centred_data_)
            
            # print(bary_x,bary_y)
            if plot_result:
                extent=(lef-bary_x,rig-bary_x-1,
                        bot-bary_y,top-bary_y-1)
                # print(extent)
                
                
                
                
                fig = plt.figure()
                if projection=='3d':
                    ax = fig.add_subplot(projection='3d')
                    ax.scatter(positions[:,1]-bary_x,
                               positions[:,0]-bary_y,
                               data_,
                               ls='-')
                else:
                    ax = fig.add_subplot()
                    ax.imshow(data_,
                                origin='lower',
                                # extent = (lef,rig,bot,top)
                                # extent=(-bary_x,width-bary_x,
                                #         -bary_y,height-bary_y),
                                extent=extent
                                )
                
                
                ax.axvline(0,c='r')
                ax.axhline(0,c='r')
                ax.set_xlim(-6,6)
                ax.set_ylim(-6,6)
                
                if fit_model:
                    npts = 101
                    X_ = np.linspace(-6,6,npts)
                    Y_ = np.linspace(-6,6,npts)
                    X, Y = np.meshgrid(X_,Y_)
                    
                    A, x0, y0, sigma_x, sigma_y, theta, zo = popt
                    fit = twoD_Gaussian((X,Y), *popt).reshape(npts,npts)
                    cmap = 'coolwarm'
                    if projection=='3d':
                        ax.plot_surface(X,Y,fit,color='C1',alpha=0.3)
                    else:
                        confidence_ellipse(ax, 
                                            x0=x0,y0=y0,
                                            sigma_x=sigma_x,
                                            sigma_y=sigma_y,
                                            theta=theta,
                                            cl=[0.68,0.95,0.99],
                                            edgecolor=color)
                        
                        
                        plt.axvline(x0,c=color)
                        plt.axhline(y0,c=color)
                    # ax.contourf(X,Y,fit,zdir = 'z', offset = -0.005, cmap=cmap)
                    # ax.contourf(X,Y,fit,zdir = 'x', offset = -6, cmap=cmap)
                    # ax.contourf(X,Y,fit,zdir = 'y', offset = 6, cmap=cmap)
                    
                
                    # ax.contour(X,Y,fit)
                    
                    
                    
            if count>=40:
                break
        else:
            continue

centred_data = np.vstack(centred_data)


    
#%%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(centred_data[...,0],
           centred_data[...,1],
           centred_data[...,2],
           marker='.')
ax.set(xlabel="Dispersion",
       ylabel="Cross-Dispersion",
       zlabel="Norm. flux",
       title=f"Segment {nseg}, curve {k}")
#%%
import numpy as np
import matplotlib.pyplot as plt

def project_and_bin(data, bin_size=2):
    """
    Projects a 3D array of shape (21, 121, 3) into a 2D pixel grid.
    
    Parameters:
        data (numpy.ndarray): 3D input array where (21, 121, 3) -> (time, points, (x, y, z)).
        bin_size (float): Size of bins for pixelation.
    
    Returns:
        numpy.ndarray: 2D grid with binned z-values.
    """
    # Average over the first axis (time dimension)
    averaged_data = np.mean(data, axis=0)  # Shape becomes (121, 3)
    
    # Extract x, y, and z coordinates
    x, y, z = averaged_data[:, 0], averaged_data[:, 1], averaged_data[:, 2]
    
    # Define grid dimensions based on data range and bin size
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_bins = np.arange(x_min, x_max, bin_size)
    y_bins = np.arange(y_min, y_max, bin_size)
    
    # Create 2D histogram grid for z-values
    grid, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins], weights=z)
    counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    
    # Normalize by counts to get the average z-value per pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        binned_z = np.where(counts > 0, grid / counts, np.nan)
    
    return binned_z, x_bins, y_bins

# Example usage
data = np.random.rand(21, 121, 3) * 100  # Example 3D array with (x, y, z) in a 100x100 space
binned_data, x_edges, y_edges = project_and_bin(centred_data, bin_size=1)

# Display the result
plt.imshow(binned_data.T, cmap='inferno', origin='lower', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
plt.colorbar(label='Averaged Z Value')
plt.title("Projected and Binned 2D Map")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#%% DO NOT USE

from scipy.signal import convolve2d

def top_hat_2d(x, y, x0, y0, wx, wy):
    return np.where((np.abs(x - x0) <= wx / 2) & (np.abs(y - y0) <= wy / 2), 1, 0)

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    return np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2)))

def convolved_model(xy, x0, y0, sigma_x, sigma_y, wx, wy, a, b, c):
    x, y = xy
    dx, dy = x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]  # Grid spacing
    
    # Generate top-hat and Gaussian
    top_hat = top_hat_2d(x, y, x0, y0, wx, wy)
    gaussian = gaussian_2d(x, y, x0, y0, sigma_x, sigma_y)
    
    # Normalize the Gaussian so that convolution maintains unit integral
    gaussian /= np.sum(gaussian) * dx * dy
    
    # Perform 2D convolution
    convolved = convolve2d(top_hat, gaussian, mode='same', boundary='fill', fillvalue=0)
    
    # Normalize amplitude
    convolved *= 1 / np.sum(convolved) * dx * dy
    
    # Position-dependent background
    background = a * x + b * y + c
    
    return convolved.ravel() + background.ravel()

def fit_convolved_model(x, y, data):
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    xy = (x_grid, y_grid)
    
    # Initial guess for parameters: x0, y0, sigma_x, sigma_y, wx, wy, a, b, c
    initial_guess = [np.mean(x), np.mean(y), 1.0, 1.0, 2.0, 2.0, 0, 0, np.mean(data)]
    
    # Perform curve fitting
    return opt.curve_fit(convolved_model, xy, data.ravel(), p0=initial_guess)
    
fit_conv_output = fit_convolved_model(
                                        centred_data[...,0],
                                        centred_data[...,1],
                                        centred_data[...,2],)
#%%


def cartesian_to_unit_circle(x,y):
    assert np.shape(x)==np.shape(y)
    r = np.hypot(x, y)
    return r / np.max(r), np.arctan2(y, x)