#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:32:59 2022

@author: dmilakov
"""

import os
import matplotlib.pyplot as plt
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

import harps.functions as hf

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.0')
pyro.set_rng_seed(0)

#%%

# note that this helper function does three different things:
# (i) plots the observed data;
# (ii) plots the predictions from the learned GP after conditioning on data;
# (iii) plots samples from the GP prior (with no conditioning on observed data)

def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):

    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.errorbar(X.numpy(), y.numpy(), e.numpy(), fmt='kx',ls='')
    if plot_predictions:
        Xtest = torch.linspace(-5.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-5.5, 5.5, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    # plt.xlim(-0.5, 5.5)
    
#%%
# N = 1000
# X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
# y = 0.5 * torch.sin(3*X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
# e = dist.Uniform(0.1, 0.2).sample(sample_shape=(N,))
X = torch.tensor(vel1s_,dtype=torch.float32)
y = torch.tensor(flx1s_,dtype=torch.float32)
e = torch.tensor(err1s_,dtype=torch.float32)
N = len(X)
plot(plot_observed_data=True)
#%%
# initialize the inducing inputs
Xu = torch.arange(50)/5-5

# initialize the kernel and model
pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1)
# kernel = gp.kernels.Sum(gp.kernels.RBF(input_dim=1), gp.kernels.RBF(input_dim=1))
# we increase the jitter for better numerical stability
sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5,
                                    # noise=e
                                     )

# the way we setup inference is similar to above
optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 2500 if not smoke_test else 2
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(sgpr.model, sgpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
# plt.plot(losses);
#%%
sgpr.set_mode('guide')
try:
    print('single')
    print('variance = {}'.format(sgpr.kernel.variance))
    print('lengthscale = {}'.format(sgpr.kernel.lengthscale))
except:
    for child in sgpr.kernel.children():
        print('child',child)
        try:
            print('variance = {}'.format(child.variance))
        except:
            pass
        try:
            print('lengthscale = {}'.format(child.lengthscale))
        except:
            pass
        try:
            print('noise = {}'.format(child.noise))
        except:
            pass
    
    # try:
    #     # print('noise = {}'.format(sgpr.kernel.noise))
    # except:
    #     pass
#%%
# let's look at the inducing points we've learned
print("inducing points:\n{}".format(sgpr.Xu.data.numpy()))
# and plot the predictions from the sparse GP
plot(model=sgpr, plot_observed_data=True, plot_predictions=True)
