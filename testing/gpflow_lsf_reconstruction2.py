#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:34:29 2022

@author: dmilakov
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
#%%
# N = 1001

# np.random.seed(0)
# tf.random.set_seed(0)

# # Build inputs X
# X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

# # Deterministic functions in place of latent ones
# f1 = np.sin
# f2 = np.cos

# # Use transform = exp to ensure positive-only scale values
# transform = np.exp

# # Compute loc and scale as functions of input X
# loc = f1(X)
# scale = transform(f2(X))

# # Sample outputs Y from Gaussian Likelihood
# Y = np.random.normal(loc, scale)
#%%
import harps.lsf as hlsf
modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,155,method='gp',subpix=10,filter=8,numpix=7,iter_solve=2)
pix3d,flx3d,err3d,orders=modeller.stack('gauss')
#%%
import harps.functions as hf
from scipy.optimize import curve_fit
od = 110
i = 15

minpix=0; maxpix=9111; numseg=16

seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
pixl = seglims[i]
pixr = seglims[i+1]

pix1s = np.ravel(pix3d[od,pixl:pixr])
flx1s = np.ravel(flx3d[od,pixl:pixr])
err1s = np.ravel(err3d[od,pixl:pixr])

X_,Y_,E_ = hlsf.clean_input(pix1s,flx1s,err1s,sort=True,verbose=True,filter=None)
X = X_[:,np.newaxis]
N = len(X)

Y = Y_[:,np.newaxis]
E = E_[:,np.newaxis] 
data = (X,Y) 

# transform = np.exp
# Compute loc and scale as functions of input X
f1 = hf.gauss3p
popt, pcov = curve_fit(f1,X_,Y_,p0=(1,1,1))

loc = f1(X_,*popt)
scale = hf.error_from_covar(f1, popt, pcov, X_)

#%%
def plot_distribution(X, Y, E, loc, scale):
    plt.figure(figsize=(15, 5))
    x = X.squeeze()
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        plt.fill_between(x, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
    plt.plot(x, lb, color="silver")
    plt.plot(x, ub, color="silver")
    plt.plot(X, loc, color="black")
    plt.errorbar(X, Y, E, color="gray", alpha=0.8, marker='.',ls='')
    plt.show()
    # plt.close()


plot_distribution(X_, Y_, E_, loc, scale)
#%%
likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")
kernel = gpf.kernels.SeparateIndependent(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        # gpf.kernels.Matern52(),
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)
# The number of kernels contained in gpf.kernels.SeparateIndependent must be the same as likelihood.latent_dim

M = 50  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
    [
        gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
)

model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)


data = (X, Y)
loss_fn = model.training_loss_closure(data)

gpf.utilities.set_trainable(model.q_mu, False)
gpf.utilities.set_trainable(model.q_sqrt, False)

variational_vars = [(model.q_mu, model.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

adam_vars = model.trainable_variables
adam_opt = tf.optimizers.Adam(0.01)


@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss_fn, variational_vars)
    adam_opt.minimize(loss_fn, adam_vars)

epochs = 200
log_freq = 20

for epoch in range(1, epochs + 1):
    optimisation_step()

    # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    if epoch % log_freq == 0 and epoch > 0:
        print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
        Ymean, Yvar = model.predict_y(X)
        Ymean = Ymean.numpy().squeeze()
        Ystd = tf.sqrt(Yvar).numpy().squeeze()
        plot_distribution(X_, Y_, E_, Ymean, Ystd)

model
    