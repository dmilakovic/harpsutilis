#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:27:41 2022

@author: dmilakov
"""

import jax
import jax.numpy as jnp
import jaxopt
from tinygp import kernels, GaussianProcess, transforms, noise
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from harps.functions import gauss4p


jax.config.update("jax_enable_x64", True)

#%%
def get_data(od,pixl,pixr,filter=50):
    import harps.lsf as hlsf
    import harps.io as io
    fname = '/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/single.dat'
    modeller=hlsf.LSFModeller(fname,60,70,method='gp',subpix=10,
                              filter=None,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    backgrounds=data['background']
    # backgrounds=None
    wavelengths=data['wavereference']
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',
                                              linelists,
                                              fluxes,
                                              wavelengths,
                                              errors,
                                              backgrounds,
                                              orders)


    # pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=True,
                                              verbose=True,filter=filter)
    
    X      = jnp.array(vel1s_)
    Y      = jnp.array(flx1s_*100)
    Y_err  = jnp.array(err1s_*100)
    # Y      = jnp.array([flx1s_,err1s_])
    return X, Y, Y_err, 
od = 100
pixl = 3000
pixr = 3500
X_,Y_,Y_err_ = get_data(100,3000,3500,None)
save_file = True
if save_file:
    filepath = f'/Users/dmilakov/software/LSF_gpmodel/data_od={od}_pix{pixl}-{pixr}.txt'
    np.savetxt(filepath, np.transpose([X_,Y_,Y_err_]))
#%%
X = X_
Y = Y_
Y_err = Y_err_

#%%
def mean_function(theta, X):
    
    gauss = jnp.exp(
        -0.5 * jnp.square((X - theta["mf_loc"]) / jnp.exp(theta["log_mf_width"]))
    )
    
    beta = jnp.array([1, gauss])
    return jnp.array([theta["mf_const"],
                      jnp.exp(theta["log_mf_amp"])/jnp.sqrt(2*jnp.pi)]) @ beta

def build_gp(theta,X, Y_err):
    
    amp   = jnp.exp(theta["log_gp_amp"])
    scale = jnp.exp(theta["log_gp_scale"])
    kernel = amp**2 * kernels.ExpSquared(scale) # LSF kernel
    
    return GaussianProcess(
        kernel,
        X,
        # diag=jnp.exp(theta['log_error']),
        noise = noise.Diagonal(Y_err**2+jnp.exp(theta['log_error'])**2),
        mean=partial(mean_function, theta),
    )

popt,pcov = curve_fit(gauss4p,X,Y,sigma=Y_err,absolute_sigma=False,p0=(1,0,1,0))
mean_params = {
    "mf_const":popt[3],
    "log_mf_amp":np.log(np.abs(popt[0])),
    "mf_loc": popt[1],
    "log_mf_width": np.log(np.abs(popt[2])),
    # "mf_linear":0.0
}
theta = dict(
    log_error = 1.0,
    log_gp_amp=np.array(1.),
    log_gp_scale=np.array(1.),
    **mean_params
)

@jax.jit
def loss(theta,X,Y,Y_err):
    gp = build_gp(theta,X, Y_err)
    return -gp.log_probability(Y)
# loss(theta)

# rng_key = jax.random.PRNGKey(0)

# from numpyro.infer.util import initialize_model
# init_params, potential_fn_gen, *_ = initialize_model(
#     rng_key,
#     model_numpyro,              # this is your numpyro model
#     model_args=(X,Y),   # add your model arguments here
#     dynamic_args=True,
# )
#%%
# solver = jaxopt.ScipyMinimize(fun=potential_fn_gen)
# solver = jaxopt.GradientDescent(fun=loss)



perr = np.sqrt(np.diag(pcov))

lbfgsb = jaxopt.ScipyBoundedMinimize(fun=partial(loss,X=X,Y=Y,Y_err=Y_err),method="l-bfgs-b")
# w_init = jnp.zeros((7,))
# lower_bounds = jnp.array([-2,-2,-2,-2,-2,-2,-2])
kappa = 10
lower_bounds = dict(
    log_error = -3.,
    log_gp_amp = -2.,
    log_gp_scale = np.log(0.4), # corresponds to 400 m/s
    log_mf_amp = np.log(np.abs(popt[0])-kappa*perr[0]),
    log_mf_width=np.log(np.abs(popt[2])-kappa*perr[2]),
    mf_const = popt[3]-kappa*perr[3],
    mf_loc = popt[1]-3*perr[1],
)
upper_bounds = dict(
    log_error = 3.0,
    log_gp_amp = 2.,
    log_gp_scale = 2.,
    log_mf_amp = np.log(np.abs(popt[0])+kappa*perr[0]),
    log_mf_width=np.log(np.abs(popt[2])+kappa*perr[2]),
    mf_const = popt[3]+kappa*perr[3],
    mf_loc = popt[1]+3*perr[1],
)
bounds = (lower_bounds, upper_bounds)
soln = lbfgsb.run(jax.tree_map(jnp.asarray, theta), bounds=bounds)
# solver = jaxopt.ScipyMinimize(fun=loss)
# soln = solver.run(jax.tree_map(jnp.asarray, theta))
print(f"Final negative log likelihood: {soln.state.fun_val}")
#%%
import harps.plotter as hplt

def plot_solution(params,X,Y,Y_err):
    X_test = jnp.linspace(X.min(), X.max(), 400)


    gp = build_gp(params,X,Y_err)
    _, cond = gp.condition(Y, X_test)
    
    mu = cond.loc
    std = np.sqrt(cond.variance)
    
    fig = hplt.Figure2(3,2, figsize=(9,6), height_ratios=[5,2,2],width_ratios=[5,1])
    
    ax1 = fig.add_subplot(0,1,0,1)
    ax2 = fig.add_subplot(1,2,0,1,sharex=ax1)
    ax3 = fig.add_subplot(2,3,0,1,sharex=ax1)
    ax4 = fig.add_subplot(2, 3, 1, 2)
    
    for ax in fig.axes:
        ax.axhline(0,ls=':',c='grey',zorder=5)
    
    # Top panel: data, full model and the gaussian model
    ax1.errorbar(X, Y, Y_err, marker='.', c='k', label="data",ls='')
    ax1.plot(X_test, mu, label="Full model",lw=2,zorder=5)
    for i in [1,3]:
        ax1.fill_between(X_test, mu + i*std, mu - i*std, color="C0", alpha=0.3)
    ax1.plot(X_test, jax.vmap(gp.mean_function)(X_test), c="C1",ls='--',
              label="Gaussian model",lw=2,zorder=4)    
    
    # Middle panel: the Gaussian process only
    _, cond_nomean = gp.condition(Y, X_test, include_mean=False)
    
    mu_nomean = cond_nomean.loc 
    std_nomean = np.sqrt(cond_nomean.variance)
    ax2.plot(X_test, mu_nomean, c='C0', ls='--', label="GP model")
    y2lims = [100,-100] # saves y limits for the middle panel
    for i in [1,3]:
        upper = mu_nomean + i*std_nomean
        lower = mu_nomean - i*std_nomean
        if np.max(lower)<y2lims[0]:
            y2lims[0]=np.min(lower)
        if np.max(upper)>y2lims[1]:
            y2lims[1]=np.max(upper)
        ax2.fill_between(X_test, upper, lower,
                          color="C0", alpha=0.3)
    # Middle panel: residuals from gaussian model
    Y_gauss_rsd = Y - jax.vmap(gp.mean_function)(X)
    ax2.scatter(X, Y_gauss_rsd, marker='.',c='grey')
    
    # Bottom left panel: normalised residuals
    _, cond_predict = gp.condition(Y, X, include_mean=True)
    mu_predict = cond_predict.loc # second term is for nicer plot
    std_predict = np.sqrt(cond_predict.variance)
    
    
    Y_tot = jnp.sqrt(std_predict**2 + Y_err**2)
    rsd = (mu_predict-Y)/Y_tot
    ax3.scatter(X,rsd,marker='.',c='grey')
    ax3_ylims = ax3.get_ylim()
    
    # Bottom right panel: a histogram of normalised residuals
    ax4.hist(np.ravel(rsd),bins=20,range=ax3_ylims,
              color='grey',orientation='horizontal',histtype='step',lw=2)
    
    
    chisq = np.sum(rsd**2)
    dof   = (len(Y)-len(params))
    labels = ['Gaussian $\mu$','Gaussian $\sigma$', 'Gaussian $A$', '$y_0$',
              'GP $\sigma$', 'GP $l$', 'log(GP error)','$N$', r'$\nu$',r'$\chi^2$',
              r'$\chi^2/\nu$','-log(probability)']
    values = [params['mf_loc'], np.exp(params['log_mf_width']), 
              np.exp(params['log_mf_amp']), params['mf_const'], 
              np.exp(params['log_gp_amp']),np.exp(params['log_gp_scale']),
              params['log_error'], len(Y),  dof, chisq, chisq/dof, loss(params,X,Y,Y_err)]
    units  = [*2*(r'kms$^{-1}$',),*3*('arb.',),'km/s',*5*('',)]
    formats = [*7*('9.3f',),*2*('5d',),*2*('9.3f',)]
    for i,(l,v,m,u) in enumerate(zip(labels,values,formats,units)):
        text = (f"{l:>20} = {v:>{m}}")
        if len(u)>0:
            text+=f' [{u}]'
        ax1.text(1.26,0.9-i*0.08,text,horizontalalignment='right',
          verticalalignment='center', transform=ax1.transAxes, fontsize=7)
        print(text)
    
    
    Xlarger = np.max([X.max(), np.abs(X.min())])
    ax1.set_xlim(-1.025*Xlarger, 1.025*Xlarger)
    y2lim = np.max([np.abs(y2lims)])
    ax2.set_ylim(-1.5*y2lim,1.5*y2lim)
    # ax1.set_ylim(-0.05,0.25)
    ax3.set_xlabel("x "+r'[kms$^{-1}$]')
    ax1.set_ylabel("y")
    ax3.set_ylabel("Norm. resids")
    ax4.set_yticklabels([])
    ax4.set_xlabel('#')
    _ = ax1.legend()
    # _ = ax2.legend()
plot_solution(soln.params,X,Y,Y_err)
#%%
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

jax.config.update("jax_enable_x64", True)


def model_numpyro_sample(x, y=None, y_err=None):
    # The parameters of the GP model
    mf_loc       = numpyro.sample("mf_loc", dist.Normal("mf_loc_mean", "mf_loc_sigma"))
    log_mf_width = numpyro.sample("log_mf_width", dist.HalfNormal(2.0))
    mf_const     = numpyro.sample("mf_const", dist.Uniform(0.0,10.0))
    log_mf_amp   = numpyro.sample("log_mf_amp", dist.HalfNormal(10.0))
    # if y_err is not None:
    #     log_gp_diag = jnp.log(y_err**2)
    # else:
    log_gp_diag  = numpyro.sample("log_gp_diag", dist.HalfNormal(1.0))
    log_gp_amp   = numpyro.sample("log_gp_amp", dist.HalfNormal(5.0))
    log_gp_scale = numpyro.sample("log_gp_scale", dist.Uniform(0.0,2.0))
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    # print(theta)
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf_ = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf_.numpyro_dist())
    
    # Intrinsic scatter
    log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
    log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
    log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0.0)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
    
def model_numpyro_param(x, y=None, y_err=None):
    # The parameters of the GP model
    mf_loc       = numpyro.param("mf_loc", 0., 
                                  constraint=dist.constraints.real)
    log_mf_width = numpyro.param("log_mf_width", 1.,
                                  constraint=dist.constraints.real)
    mf_const     = numpyro.param("mf_const", 0., 
                                  constraint=dist.constraints.real)
    log_mf_amp   = numpyro.param("log_mf_amp", 0., 
                                  constraint=dist.constraints.real)
    # if y_err is not None:
    #     log_gp_diag = jnp.log(y_err**2)
    # else:
    log_gp_diag  = numpyro.param("log_gp_diag", 1e-5, 
                                  constraint=dist.constraints.real)
    log_gp_amp   = numpyro.param("log_gp_amp", 0., 
                                  constraint=dist.constraints.real)
    log_gp_scale = numpyro.param("log_gp_scale", 0., 
                                  constraint=dist.constraints.real)
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    # print(theta)
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf_ = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf_.numpyro_dist())
    
    # Intrinsic scatter
    log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
    log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
    log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0.0)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
    
def guide_numpyro(x, y=None, y_err=None):
    # The parameters of the GP model
    mf_loc       = numpyro.param("mf_loc_mean", 0., 
                                 constraint=dist.constraints.real)
    mf_loc       = numpyro.param("mf_loc_sigma", 0., 
                                 constraint=dist.constraints.real)
    log_mf_width = numpyro.param("log_mf_width", 1.,
                                  constraint=dist.constraints.real)
    mf_const     = numpyro.param("mf_const", 0., 
                                 constraint=dist.constraints.real)
    log_mf_amp   = numpyro.param("log_mf_amp", 0., 
                                 constraint=dist.constraints.real)
    # if y_err is not None:
    #     log_gp_diag = jnp.log(y_err**2)
    # else:
    log_gp_diag  = numpyro.param("log_gp_diag", 1e-5, 
                                 constraint=dist.constraints.real)
    log_gp_amp   = numpyro.param("log_gp_amp", 0., 
                                 constraint=dist.constraints.real)
    log_gp_scale = numpyro.param("log_gp_scale", 0., 
                                 constraint=dist.constraints.real)
    # sys.exit()
    
    # diag = 1e-5
    theta = dict(
        mf_loc=mf_loc,
        log_mf_width=log_mf_width,
        mf_const=mf_const,
        log_mf_amp=log_mf_amp,
        log_gp_diag=log_gp_diag,
        log_gp_amps=log_gp_amp,
        log_gp_scale=log_gp_scale,
    )
    # Set up the kernel and GP objects for the LSF
    # print(theta)
    kernel_lsf = jnp.exp(log_gp_amp) * kernels.ExpSquared(log_gp_scale)
    gp_lsf_ = GaussianProcess(kernel_lsf, 
                             x, 
                             diag=log_gp_diag, 
                             mean=partial(mean_function, theta))
    
    # LSF model
    inferred_lsf = numpyro.sample("gp_lsf", gp_lsf_.numpyro_dist())
    
    # Intrinsic scatter
    log_sct_amp   = numpyro.sample("log_sct_amp", dist.HalfNormal(5.0))
    log_sct_scale = numpyro.sample("log_sct_scale", dist.Uniform(0.0,2.))
    log_sct_diag  = numpyro.sample("log_sct_diag", dist.Uniform(0.0,2.0))
    kernel_sct    = jnp.exp(log_sct_amp) * kernels.ExpSquared(log_sct_scale)
    
    gp_scatter = GaussianProcess(kernel_sct, 
                                 x, 
                                 diag=log_sct_diag, 
                                 mean=0.0)
    
    # log scatter model
    log_inferred_sct = numpyro.sample("gp_sct", gp_scatter.numpyro_dist())
    
    numpyro.sample("obs", 
                   dist.Normal(inferred_lsf, jnp.exp(log_inferred_sct)),
                   obs=y)
#%%
# Run the MCMC
nuts_kernel = numpyro.infer.NUTS(model_numpyro_sample, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=200,
    num_samples=200,
    num_chains=2,
    progress_bar=True,
)
rng_key = jax.random.PRNGKey(55873)
mcmc.run(rng_key, X, y=Y, y_err=Y_err)
samples = mcmc.get_samples()
#%%
q = np.percentile(samples["gp_lsf"], [5, 25, 50, 75, 95], axis=0)
plt.plot(X, q[2], color="C0", label="MCMC inferred model")
plt.fill_between(X, q[0], q[-1], alpha=0.3, lw=0, color="C0")
plt.fill_between(X, q[1], q[-2], alpha=0.3, lw=0, color="C0")
# plt.plot(X, true_log_rate, "--", color="C1", label="true rate")
plt.errorbar(X, Y, Y_err, ls='', marker='.', c='k', label="data")
plt.legend(loc=2)
plt.xlabel("x")
_ = plt.ylabel("counts")
#%%
import arviz as az

data = az.from_numpyro(mcmc)
print(az.summary(
    data, var_names=[v for v in data.posterior.data_vars if v != "pred"]
))
#%% Plot corner plot
data=az.from_dict(samples)
var_names = ["mf_loc", "log_mf_width", "mf_const", "log_mf_amp",
             "log_gp_amp","log_gp_scale","log_sct_amp","log_sct_scale"]
az.plot_pair(
    data,
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)
#%%  SVI using Autodelta
from numpyro.infer.autoguide import AutoDelta
guide = AutoDelta(model_numpyro_sample)
optim = numpyro.optim.Adam(0.01)
svi = numpyro.infer.SVI(model_numpyro_sample, guide, optim, numpyro.infer.Trace_ELBO(10))
results = svi.run(jax.random.PRNGKey(55873), 3000, X, y=Y, progress_bar=True)
#%%


#%% Latent kernel

class LatentKernel(kernels.Kernel):
    def __init__(self, kernel1,kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        # self.theta  = theta
    def evaluate(self, X1, X2):
        # label=0 for observed data, label=1 for observed noise
        x1, label1 = X1 
        x2, label2 = X2 
        

        # Evaluate the kernel matrix and all of its relevant derivatives
        K1 = self.kernel1.evaluate(x1, x2)
        K2 = self.kernel2.evaluate(x1, x2)
        

        return K1+K2

#%%
def gauss4p(x, amplitude, center, sigma, y0 ):
    # Four parameters: amplitude, center, width, y-offset
    #y = np.zeros_like(x,dtype=np.float64)
    #A, mu, sigma, y0 = p
    y = y0+ amplitude/jnp.sqrt(2*np.pi)/sigma*jnp.exp((-((x-center)/sigma)**2)/2)
    return y
def model_gauss(x, y=None, y_err=None):
    loc   = numpyro.sample('loc',dist.Normal(0.0,2.0))
    sigma = numpyro.sample('sigma',dist.HalfNormal(2*np.std(x)))
    
    if y is not None:
        amp   = numpyro.sample('amp',dist.Normal(np.max(y),2*np.std(y)))
        offset = numpyro.sample('offset',dist.Normal(0.,0.1*np.std(y)))
    else:
        amp = numpyro.sample('amp'.dist.Uniform(*np.percentile(y,[50,99])))
        offset = numpyro.sample('offset',dist.Uniform(0.,1.))
    if y_err is not None:
        err = numpyro.sample('err',dist.Normal(0.,3*np.std(y_err)))
    else:
        err = numpyro.sample('err',dist.Normal(0.,1.))
    fun = jnp.array(gauss4p(x,amp,loc,sigma,offset))
    numpyro.sample('obs',dist.Normal(fun,err),obs=y)
    
nuts_kernel = numpyro.infer.NUTS(model_gauss, target_accept_prob=0.9)
mcmc = numpyro.infer.MCMC(
    nuts_kernel,
    num_warmup=1000,
    num_samples=1000,
    num_chains=4,
    progress_bar=True,
)
rng_key = jax.random.PRNGKey(55873)
mcmc.run(rng_key, X, y=Y, y_err=Y_err)
samples = mcmc.get_samples()
#%% Plot corner plot
import arviz as az
data=az.from_dict(samples)
var_names = ["loc","sigma","amp","offset"]
az.plot_pair(
    data,
    var_names=var_names,
    kind="kde",
    marginals=True,
    point_estimate="median",
    divergences=True,
    textsize=12,
)