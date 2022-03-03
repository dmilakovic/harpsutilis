#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:10:21 2022

@author: dmilakov
"""

import itertools
import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
import harps.io as hio
import harps.lsf as hlsf

rng = np.random.RandomState(123)
tf.random.set_seed(42)


#%%
modeller=hlsf.LSFModeller('/Users/dmilakov/projects/lfc/dataprod/output/v_1.2/test.dat',60,155,method='gp',subpix=10,filter=8,numpix=7,iter_solve=2)
pix3d,flx3d,err3d,orders=modeller.stack('gauss')
#%%
od = 100
i = 5

minpix=0; maxpix=9111; numseg=16

seglims = np.linspace(minpix,maxpix,numseg+1,dtype=int)
pixl = seglims[i]
pixr = seglims[i+1]

pix1s = np.ravel(pix3d[od,pixl:pixr])
flx1s = np.ravel(flx3d[od,pixl:pixr])
err1s = np.ravel(err3d[od,pixl:pixr])

X,Y,E = hlsf.clean_input(pix1s,flx1s,err1s,sort=True,verbose=True,filter=None)
X = X[:,np.newaxis]
N = len(X)

Y = Y[:,np.newaxis] #+ 0.02*rng.randn(N, 1) 
data = (X,Y) 

M = rng.randint(0,N,size=50)  # Filtering. Number of inducing locations = N//M
# M = rng.normal(N//2,np.std(X),N)*100

linear = gpflow.kernels.SquaredExponential()
linear.lengthscales.assign(2)
noise  = gpflow.kernels.White()
noise.variance.assign(np.mean(E))
periodic = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
periodic.base_kernel.lengthscales.assign(5)
periodic.period.assign(10)
matern = gpflow.kernels.Matern52()

kernel = matern # linear #+ noise + periodic

# kernel = gpflow.kernels.SquaredExponential()

Z = X[M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

#%%
plt.plot(X, Y, "x", alpha=0.2)
plt.plot(X[M],Y[M],'o',alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
# Yt = func(Xt)
# _ = plt.plot(Xt, Yt, c="k")
#%%

#%%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


N = 10000  # Number of training observations

X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

M = 50
kernel0 = gpflow.kernels.SquaredExponential()
periodic = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
periodic.period.assign(8)

kernel = periodic
Z = X[:M, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=N)

#%%
X1 = np.array([[0.0]])
X2 = np.linspace(-10, 10, 1001).reshape(-1, 1)

K21 = kernel(X2, X1)  # cov(f(X2), f(X1)): matrix with shape [101, 1]
K22 = kernel(X2)  # equivalent to k(X2, X2) (but more efficient): matrix with shape [101, 101]

# plotting
plt.figure()
_ = plt.plot(X2, K21)

#%%
elbo = tf.function(m.elbo)

# TensorFlow re-traces & compiles a `tf.function`-wrapped method at *every* call if the arguments are numpy arrays instead of tf.Tensors. Hence:
tensor_data = tuple(map(tf.convert_to_tensor, data))
elbo(tensor_data)  # run it once to trace & compile
#%%
minibatch_size = 100

train_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).repeat().shuffle(N)

train_iter = iter(train_dataset.batch(minibatch_size))

ground_truth = elbo(tensor_data).numpy()
#%%
evals = [elbo(minibatch).numpy() for minibatch in itertools.islice(train_iter, 100)]

plt.hist(evals, label="Minibatch estimations")
plt.axvline(ground_truth, c="k", label="Ground truth")
plt.axvline(np.mean(evals), c="g", ls="--", label="Minibatch mean")
plt.legend()
plt.title("Histogram of ELBO evaluations using minibatches")
print("Discrepancy between ground truth and minibatch estimate:", ground_truth - np.mean(evals))

#%%
def plot(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    pX = np.linspace(np.min(X), np.max(X), 500)[:, None]  # Test locations
    pY, pYv = m.predict_y(pX)  # Predict Y values at test locations
    plt.plot(X, Y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(pX, pY, lw=1.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        pX[:, 0],
        (pY - 2 * pYv ** 0.5)[:, 0],
        (pY + 2 * pYv ** 0.5)[:, 0],
        color=col,
        alpha=0.6,
        lw=1.5,
    )
    Z = m.inducing_variable.Z.numpy()
    plt.plot(Z, np.zeros_like(Z), "k|", mew=2, label="Inducing locations")
    plt.legend(loc="lower right")


plot(title="Predictions before training")
#%%
minibatch_size = 100

# We turn off training for inducing point locations
gpflow.set_trainable(m.inducing_variable, False)


def run_adam(model, iterations):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 10 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
    return logf
#%%
maxiter = ci_niter(20000)

logf = run_adam(m, maxiter)
plt.plot(np.arange(maxiter)[::10], logf)
plt.xlabel("iteration")
_ = plt.ylabel("ELBO")
#%%
plot("Predictions after training")