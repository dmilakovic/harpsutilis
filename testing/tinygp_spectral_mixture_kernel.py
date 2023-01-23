#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:02:09 2022

@author: dmilakov
"""
import tinygp
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tinygp import noise, kernels
#%%
import harps.lsf as hlsf
modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat',
                          60,101,method='tinygp',
                          subpix=10,filter=2,numpix=8,iter_solve=2,iter_center=2)
wavelengths = modeller['wavereference']
fluxes      = modeller['flux']
backgrounds = modeller['background']
errors      = modeller['error']
linelists   = modeller['linelist']
fittype     = 'gauss'
pix3d,vel3d,flx3d,err3d,orders = hlsf.stack(fittype,linelists,fluxes,
                                    wavelengths,errors,backgrounds,
                                    modeller._orders)
#%%
od=100
seg = 5
pixl=9111//16*seg
pixr=9111//16*(seg+1)

pix1s=pix3d[od,pixl:pixr]#[:,0]
vel1s=vel3d[od,pixl:pixr]#[:,0]
flx1s=flx3d[od,pixl:pixr]#[:,0]
err1s=err3d[od,pixl:pixr]#[:,0]

rng_key=jax.random.PRNGKey(55825) # original
# rng_key=jax.random.PRNGKey(55826)
# rng_key=jax.random.PRNGKey(558257)
# rng_key=jax.random.PRNGKey(55822)
# rng_key=jax.random.PRNGKey(558214)
x1s_, flx1s_, err1s_ = hlsf.clean_input(
                                        pix1s,
                                        # vel1s,
                                        flx1s,err1s,sort=True,
                                          rng_key=rng_key,
                                          verbose=True,filter=None)
X = jnp.array(x1s_)
Y = jnp.array(flx1s_)
Y_err = jnp.array(err1s_)
#%%

class SpectralMixture(tinygp.kernels.Kernel):
    def __init__(self, weight, scale, freq):
        self.weight = jnp.atleast_1d(weight)
        self.scale = jnp.atleast_1d(scale)
        self.freq = jnp.atleast_1d(freq)

    def evaluate(self, X1, X2):
        tau = jnp.atleast_1d(jnp.abs(X1 - X2))[..., None]
        return jnp.sum(
            self.weight
            * jnp.prod(
                jnp.exp(-2 * jnp.pi**2 * tau**2 / self.scale**2)
                * jnp.cos(2 * jnp.pi * self.freq * tau),
                axis=-1,
            )
        )
    def spectral_density(self,s):
        
        return jnp.sum(
            # self.weight * 0.5 * jnp.sqrt(self.scale **2 / (2*jnp.pi)) 
            #     * jnp.exp(-self.scale**2 * (s**2 + 1./self.freq**2)),
            self.weight * 0.5 * jnp.sqrt(self.scale **2 / (2*jnp.pi)) 
                * jnp.exp(-self.scale**2 * (1./s - 1./self.freq)**2),
              axis = -1,
              )
def build_gp(theta,x,y_err):
    kernel = SpectralMixture(
        jnp.exp(theta["log_weight"]),
        jnp.exp(theta["log_scale"]),
        jnp.exp(theta["log_freq"]),
    )
    add_var = jnp.exp(theta["log_diag"])
    obs_var = y_err**2
    tot_var = add_var + obs_var
    diag = noise.Diagonal(tot_var)
    return tinygp.GaussianProcess(
        kernel, x, noise=diag, mean=theta["mean"]
    )

#%%
params = {
    "log_weight": np.log([30.0, 1.0, 1.0, 1.0, 2.0, 5.0]),
    "log_scale": np.log([0.3, 2.0, 3.0, 1.0, 3.0, 2.0]),
    "log_freq": np.log([0.03, 0.06, 0.0001, 0.09, 0.2, 0.002]),
    "log_diag": np.log(0.1),
    "mean": 0.0,
}
init_params = params
true_gp = build_gp(init_params,X,Y_err)
#%%
#random = np.random.default_rng(546)
# t = np.sort(random.uniform(0, 10, 50))
# t = X
# y_err = Y_err
# y = true_gp.sample(jax.random.PRNGKey(123))
# y = Y

#%%

import optax


@jax.jit
@jax.value_and_grad
def loss(theta,X,Y,Y_err):
    return -build_gp(theta,X,Y_err).log_probability(Y)


opt = optax.sgd(learning_rate=3e-4)
opt_state = opt.init(params)
for i in range(1000):
    loss_val, grads = loss(params,X,Y,Y_err)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
print(params)
fit_params = params
#%%

# plt.plot(t, y, ".k")
# # plt.ylim(-4.5, 4.5)
# plt.title("simulated data")
# plt.xlabel("x")
# _ = plt.ylabel("y")
plt.figure()
x = np.linspace(X.min(), X.max(), 500)
plt.errorbar(X, Y,Y_err, marker='o',ms=2,c='k', ls='',label="data")
true_gp = build_gp(init_params,X,Y_err)
gp_init = true_gp.condition(Y,x).gp
mu_init, var_init = gp_init.loc, gp_init.variance
# plt.fill_between(
#     x,
#     mu_init + np.sqrt(var_init),
#     mu_init - np.sqrt(var_init),
#     color="C1",
#     alpha=0.5,
#     label="Initial",
# )
# plt.plot(x, mu_init, color="C1", lw=2,label="Initial",)

opt_gp = build_gp(fit_params,X,jnp.zeros_like(Y))
gp_cond = opt_gp.condition(Y, x).gp
mu, var = gp_cond.loc, gp_cond.variance
plt.fill_between(
    x,
    mu + np.sqrt(var),
    mu - np.sqrt(var),
    color="C0",
    alpha=0.5,
    label="conditional",
)
plt.plot(x, mu, color="C0", lw=2, label="Conditional",)
# plt.xlim(x.min(), x.max())
# plt.ylim(-4.5, 4.5)
plt.legend(loc=2)
plt.xlabel("x")
_ = plt.ylabel("y")

plt.xlim(-5,-2); plt.ylim(-1,1)