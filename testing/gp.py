# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Gaussian Process
=========================

In this example we show how to use NUTS to sample from the posterior
over the hyperparameters of a gaussian process.

.. image:: ../_static/img/examples/gp.png
    :align: center
"""

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)

# matplotlib.use("Agg")  # noqa: E402


# squared exponential kernel with diagonal noise term
def kernel(X, Z, var1, length1, var2, length2, noise, error, 
           jitter=1.0e-6, include_noise=True):
    deltaXsq1 = jnp.power((X[:, None] - Z) / length1, 2.0)
    k1 = var1 * jnp.exp(-0.5 * deltaXsq1)
    deltaXsq2 = jnp.power((X[:, None] - Z) / length2, 2.0)
    k2 = var2 * jnp.exp(-0.5 * deltaXsq2)
    k  = k1 + k2
    if include_noise:
        k += (noise + error + jitter) * jnp.eye(X.shape[0])
    return k


def model(X, Y, Y_err):
    # set uninformative log-normal priors on our three kernel hyperparameters
    var1 = numpyro.sample("kernel_var1", dist.LogNormal(0.0, 10.0))
    length1 = numpyro.sample("kernel_length1", dist.LogNormal(0.0, 10.0))
    var2 = numpyro.sample("kernel_var2", dist.LogNormal(0.0, 10.0))
    length2 = numpyro.sample("kernel_length2", dist.LogNormal(0.0, 10.0))
    
    noise = numpyro.sample("kernel_noise", dist.LogNormal(0.0, 10.0))
    

    # compute kernel
    k = kernel(X, X, var1, length1, var2, length2, noise, Y_err)

    # sample Y according to the standard gaussian process formula
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
        obs=Y,
    )


# helper function for doing hmc inference
def run_inference(model, args, rng_key, X, Y, Y_err):
    start = time.time()
    # demonstrate how to use different HMC initialization strategies
    if args.init_strategy == "value":
        init_strategy = init_to_value(
            values={"kernel_var1": 1.0, "kernel_length1": 0.5, 
                    "kernel_var2": 1.0, "kernel_length2": 0.5, 
                    "kernel_noise": 0.05, }
        )
    elif args.init_strategy == "median":
        init_strategy = init_to_median(num_samples=10)
    elif args.init_strategy == "feasible":
        init_strategy = init_to_feasible()
    elif args.init_strategy == "sample":
        init_strategy = init_to_sample()
    elif args.init_strategy == "uniform":
        init_strategy = init_to_uniform(radius=1)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        thinning=args.thinning,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, Y_err)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# do GP prediction for a given set of hyperparameters. this makes use of the well-known
# formula for gaussian process predictions
def predict(rng_key, X, Y, Y_err, X_test, var1, length1, var2, length2, noise):
    # compute kernels between train and test data, etc.
    k_pp = kernel(X_test, X_test, var1, length1, var2, length2, noise, Y_err, include_noise=True)
    k_pX = kernel(X_test, X, var1, length1, var2, length2,noise, Y_err, include_noise=False)
    k_XX = kernel(X, X, var1, length1, var2, length2, noise, Y_err, include_noise=True)
    K_xx_inv = jnp.linalg.inv(k_XX)
    K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )
    mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


# create artificial regression dataset
# def get_data(N=30, sigma_obs=0.15, N_test=400):
#     np.random.seed(0)
#     X = jnp.linspace(-1, 1, N)
#     Y = X + 0.2 * jnp.power(X, 3.0) + 0.5 * jnp.power(0.5 + X, 2.0) * jnp.sin(4.0 * X)
#     Y += sigma_obs * np.random.randn(N)
#     Y -= jnp.mean(Y)
#     Y /= jnp.std(Y)

#     assert X.shape == (N,)
#     assert Y.shape == (N,)

#     X_test = jnp.linspace(-1.3, 1.3, N_test)

#     return X, Y, X_test
#%%
def get_data(od,N_test=400):
    import harps.lsf as hlsf
    import harps.io as io
    modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,70,method='gp',subpix=10,filter=10,numpix=8,iter_solve=1,iter_center=1)

    extensions = ['linelist','flux','background','error','wavereference']
    data, numfiles = io.mread_outfile(modeller._outfile,extensions,701,
                            start=None,stop=None,step=None)
    linelists=data['linelist']
    fluxes=data['flux']
    errors=data['error']
    # backgrounds=data['background']
    backgrounds=None
    wavelengths=data['wavereference']
    orders=np.arange(od,od+1)
    pix3d,vel3d,flx3d,err3d,orders=hlsf.stack('gauss',linelists,fluxes,wavelengths,errors,backgrounds,orders)
    pixl=5000
    pixr=5500

    # pix1s=pix3d[od,pixl:pixr]
    vel1s=vel3d[od,pixl:pixr]
    flx1s=flx3d[od,pixl:pixr]
    err1s=err3d[od,pixl:pixr]

    vel1s_, flx1s_, err1s_ = hlsf.clean_input(vel1s,flx1s,err1s,sort=True,verbose=True,filter=100)
    
    X      = jnp.array(vel1s_)
    Y      = jnp.array(flx1s_)
    Y_err  = jnp.array(err1s_)
    # Y      = jnp.array([flx1s_,err1s_])
    X_test = jnp.linspace(-5.5, 5.5, N_test)
    return X, Y, Y_err, X_test
#%%

def main(args):
    X, Y, Y_err, X_test = get_data(od=args.order)

    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, Y_err)

    # do prediction
    vmap_args = (
        random.split(rng_key_predict, samples["kernel_var1"].shape[0]),
        samples["kernel_var1"],
        samples["kernel_length1"],
        samples["kernel_var2"],
        samples["kernel_length2"],
        samples["kernel_noise"],
    )
    means, predictions = vmap(
        lambda rng_key, var1, length1, var2, length2, noise: predict(
            rng_key, X, Y, Y_err, X_test, var1, length1, var2, length2, noise
        )
    )(*vmap_args)

    mean_prediction = np.mean(means, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)

    # make plots
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # plot training data
    ax.plot(X, Y, "kx")
    # plot 90% confidence level of predictions
    ax.fill_between(X_test, percentiles[0, :], percentiles[1, :], color="lightblue")
    # plot mean prediction
    ax.plot(X_test, mean_prediction, "blue", ls="solid", lw=2.0)
    ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    plt.savefig("gp_plot.pdf")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.9.1")
    parser = argparse.ArgumentParser(description="Gaussian Process example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    parser.add_argument("-od", "--order", nargs="?", default=100, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--thinning", nargs="?", default=2, type=int)
    parser.add_argument("--num-data", nargs="?", default=25, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument(
        "--init-strategy",
        default="median",
        type=str,
        choices=["median", "feasible", "value", "uniform", "sample"],
    )
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
