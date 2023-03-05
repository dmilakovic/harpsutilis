#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:41:56 2023

Can I sum the GPs and condition the summed GP once?

@author: dmilakov
"""

import harps.lsf.gp_aux as gp_aux
import matplotlib.pyplot as plt
import numpy as np
#%%
from fitsio import FITS
filename = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits'
filename = '/Users/dmilakov/projects/lfc/dataprod/lsf/v_1.2/HARPS.2018-12-05T08:12:52.040_e2ds_A_lsf.fits_bk'
hdul=FITS(filename)
lsf1d50=hdul[1].read()
#%%
bary = 130.2245
LSF_data, weights=gp_aux.extract_lists('LSF',bary,lsf1d50,N=2)
#%%
import harps.lsf.gp as gp
import jax.numpy as jnp
import scipy.linalg
import tinygp

class MyKernel(tinygp.kernels.Kernel):
    def __init__(self, kernel1, kernel2, weights):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.weights = weights
        # self.kernel  = scipy.linalg.block_diag(kernel1,kernel2)

    def evaluate(self, X1, X2):
        return self.weights[0]*self.kernel1.evaluate(X1,X2) +\
               self.weights[1]*self.kernel2.evaluate(X1,X2) 
               
gp1=gp.build_LSF_GP(*LSF_data[0])
gp2=gp.build_LSF_GP(*LSF_data[1])
kernel1=gp1.kernel#.kernel2
kernel2=gp2.kernel#.kernel2
# w1,w2 = np.flip(weights)
# w1,w2 = weights #* jnp.sqrt(2*jnp.pi)*scales**2
w1, w2 = (1,0)
# kernel_sum=w1*kernel1+w2*kernel2
kernel_comb = MyKernel(kernel1,kernel2,weights)
 
x1=gp1.X
x2=gp2.X
Y1=LSF_data[0][2]
Y2=LSF_data[1][2]
Ye1=LSF_data[0][3]/w1
Ye2=LSF_data[1][3]/w2



X_comb_=jnp.hstack([x1,x2])
sorter = jnp.argsort(X_comb_)
X_comb = X_comb_[sorter]
Y_comb=jnp.hstack([Y1,Y2])[sorter]
Ye_comb=jnp.hstack([Ye1,Ye2])[sorter]

# noise1=gp.noise.Dense(gp1.noise.value/w1)#*np.sqrt(w1))
# noise2=gp.noise.Dense(gp2.noise.value/w2)#*np.sqrt(w2))
# noise_comb=scipy.linalg.block_diag(noise1.value,noise2.value)

gp_comb=gp.GaussianProcess(kernel_comb,X=X_comb,noise=gp.noise.Diagonal(Ye_comb))

x_test=jnp.linspace(X_comb.min(),X_comb.max(),300)
gp_cond=gp_comb.condition(Y_comb,x_test)
gp_cond
_,gp_cond=gp_comb.condition(Y_comb,x_test)
mean=gp_cond.mean
sigma=jnp.sqrt(gp_cond.variance)

#%%
plt.figure()
plt.errorbar(x1,Y1,Ye1,marker='o',ls='',capsize=2,color='C2')
plt.errorbar(x2,Y2,Ye2,marker='o',ls='',capsize=2,color='C3')
# plt.errorbar(X_comb,Y_comb,Ye_comb,marker='o',ls='',capsize=2)
plt.plot(x_test,mean,zorder=10,label=f'{w1:1.2f}*GP1+{w2:1.2f}*GP2',c='C1')
plt.fill_between(x_test,mean+sigma,mean-sigma,color='C1',alpha=0.3)
for i,(gp_,Y_) in enumerate(zip([gp1,gp2],[Y1,Y2])):
    _, gp_cond_ = gp_.condition(Y_,x_test)
    mean_ = gp_cond_.mean
    sigma_ = jnp.sqrt(gp_cond_.variance)
    plt.plot(x_test,mean_,zorder=10,label=f'GP{i+1}',c=f'C{i+2}')
    plt.fill_between(x_test,mean_+sigma_,mean_-sigma_,color=f'C{i+2}',alpha=0.3)
# plt.xlim(-5,-3)
# plt.ylim(-1.5,1.5)
plt.legend()
#%%#%%
def plot_kernel(kernel, **kwargs):
    dx = np.linspace(0, 5, 100)
    plt.plot(dx, kernel(dx, dx[:1]), **kwargs)
    plt.xlabel("dx")
    plt.ylabel("k(dx)")
plt.figure()
plot_kernel(kernel1, label='GP1')
plot_kernel(w1*kernel1, label=f'{w1:1.3f}*GP1')
plot_kernel(kernel2, label='GP2')
plot_kernel(w2*kernel2, label=f'{w2:1.3f}*GP2')
plot_kernel(kernel1+kernel2, label='GP1+GP2')
plot_kernel(w1*kernel1+w2*kernel2, label=f'{w1:1.3f}*GP1+{w2:1.3f}*GP2')
plt.legend()
#%%
import jax
import tinygp 
from tinygp.helpers import JAXArray, dataclass, field
from tinygp.solvers.solver import Solver
from tinygp.solvers.direct import DirectSolver
from tinygp.kernels.base import Kernel
@dataclass
class MyGP(Kernel):
    kernel1: Kernel
    kernel2: Kernel
    weights: JAXArray
    X1: JAXArray
    X2: JAXArray
    noise1: JAXArray
    noise2: JAXArray
    solver1: DirectSolver.init(
            kernel1, X1, noise1, covariance=None
        )
    solver2: DirectSolver.init(
            kernel2, X2, noise2, covariance=None
        )


    def evaluate_kernel1(self,  X1: JAXArray, X2: JAXArray) -> JAXArray:
        kernel_vec = jax.vmap(self.kernel1.evaluate, in_axes=(0, None))
        K1 = self.solver.solve_triangular(kernel_vec(self.X, X1))
        K2 = self.solver.solve_triangular(kernel_vec(self.X, X2))
        return self.kernel1.evaluate(X1, X2) - K1.transpose() @ K2
    def evaluate_kernel2(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        kernel_vec = jax.vmap(self.kernel2.evaluate, in_axes=(0, None))
        K1 = self.solver.solve_triangular(kernel_vec(self.X, X1))
        K2 = self.solver.solve_triangular(kernel_vec(self.X, X2))
        return self.kernel2.evaluate(X1, X2) - K1.transpose() @ K2
    def evaluate(self, X1: JAXArray, X2: JAXArray) -> JAXArray:
        eval1 = self.evaluate_kernel1(X1, X2)
        eval2 = self.evaluate_kernel2(X1, X2)
        return self.weights[0]*eval1 + self.weights[1]*eval2
    def evaluate_diag_kernel1(self, X: JAXArray) -> JAXArray:
        kernel_vec = jax.vmap(self.kernel1.evaluate, in_axes=(0, None))
        K = self.solver.solve_triangular(kernel_vec(self.X, X))
        return self.kernel1.evaluate_diag(X) - K.transpose() @ K
    def evaluate_diag_kernel2(self, X: JAXArray) -> JAXArray:
        kernel_vec = jax.vmap(self.kernel2.evaluate, in_axes=(0, None))
        K = self.solver.solve_triangular(kernel_vec(self.X, X))
        return self.kernel2.evaluate_diag(X) - K.transpose() @ K
    def evaluate_diag(self, X: JAXArray) -> JAXArray:
        eval1 = self.evaluate_diag_kernel1(X)
        eval2 = self.evaluate_diag_kernel2(X)
        return self.weights[0]*eval1 + self.weights[1]*eval2


gp1=gp.build_LSF_GP(*LSF_data[0])
gp2=gp.build_LSF_GP(*LSF_data[1])
kernel1=gp1.kernel#.kernel2
kernel2=gp2.kernel#.kernel2
x1=LSF_data[0][1]
x2=LSF_data[1][1]
Y1=LSF_data[0][2]
Y2=LSF_data[1][2]
Ye1=LSF_data[0][3]#/w1
Ye2=LSF_data[1][3]#/w2
# w1,w2 = np.flip(weights)
# w1,w2 = weights 
w1, w2 = (1,0)
x_test=jnp.linspace(X_comb.min(),X_comb.max(),300)

kernel_comb = MyKernel2(kernel1,kernel2,weights,x1,x2,Ye1,Ye2)
#%%
 
x1=LSF_data[0][1]
x2=LSF_data[1][1]
Y1=LSF_data[0][2]
Y2=LSF_data[1][2]
Ye1=LSF_data[0][3]#/w1
Ye2=LSF_data[1][3]#/w2



X_comb_=jnp.hstack([x1,x2])
sorter = jnp.argsort(X_comb_)
X_comb = X_comb_[sorter]
Y_comb=jnp.hstack([Y1,Y2])[sorter]
Ye_comb=jnp.hstack([Ye1,Ye2])[sorter]

# noise1=gp.noise.Dense(gp1.noise.value/w1)#*np.sqrt(w1))
# noise2=gp.noise.Dense(gp2.noise.value/w2)#*np.sqrt(w2))
# noise_comb=scipy.linalg.block_diag(noise1.value,noise2.value)
# gp_comb=gp.GaussianProcess(kernel_comb,X=X_comb,noise=gp.noise.Diagonal(Ye_comb))

x_test=jnp.linspace(X_comb.min(),X_comb.max(),300)

_,gp_cond=gp_comb.condition(Y_comb,x_test)
mean=gp_cond.mean
sigma=jnp.sqrt(gp_cond.variance)
