from scipy.special import gamma
from scipy.special import kv
import numpy as np
import matplotlib.pyplot as plt
import os

def setup_env():
    global L, NU
    L = 4
    NU = 8

def sqexp_kernel_func(x1_grid_G, x2_grid_G):
   cov_list = []
   for x in x1_grid_G:
       for x_prime in x2_grid_G:
            cov_list.append(np.exp(- ((x - x_prime) ** 2)/(2 * (L ** 2))))
   cov_func = np.array(cov_list).reshape(len(x1_grid_G), len(x2_grid_G))
   return cov_func



def matern_kernel_func(x1_grid_G, x2_grid_G):
    cov_list = []
    for x in x1_grid_G:
        for x_prime in x2_grid_G:
            cov_list.append(((2 ** (1 - NU)) / gamma(NU)) * (((np.sqrt(2 * NU) * np.maximum(np.abs(x - x_prime), 1e-15)) / L) ** NU)\
                            * kv(NU, (np.sqrt(2 * NU) * np.maximum(np.abs(x - x_prime), 1e-15) / L)))
    cov_func = np.array(cov_list).reshape(len(x1_grid_G), len(x2_grid_G))
    return cov_func

def draw_GP_prior_samples_at_x_grid(
        x_grid_G, mean, cov_func,
        random_seed=42,
        n_samples=1):
    """ Draw sample from GP prior given mean/cov functions
    Args
    ----
    Returns
    -------
    f_SG : 2D array, n_samples (S) x n_grid_pts (G)
        Contains sampled function values at each point of x_grid
    """
    prior_samples = []
    # Use consistent random number generator for reproducibility
    cov_arr = cov_func(x_grid_G, x_grid_G)
    prng = np.random.RandomState(int(random_seed))
    for _ in range(n_samples):
        prior_samples.append(prng.multivariate_normal(mean, cov_arr))
    prior_samples = np.array(prior_samples).reshape(n_samples, len(x_grid_G))
    return prior_samples


def draw_GP_posterior_samples_at_x_grid(
        x_train_N, y_train_N, x_grid_G, mean, cov_func,
        sigma=0.1,
        random_seed=42,
        n_samples=1):
    """ Draw sample from GP posterior given training data and mean/cov

    Args
    ----
    Returns
    -------
    f_SG : 2D array, n_samples (S) x n_grid_pts (G)
        Contains sampled function values at each point of x_grid
    """
    posterior_samples = []
    # Use consistent random number generator for reproducibility
    new_mean = cov_func(x_grid_G, x_train_N).dot(np.linalg.inv(cov_func(x_train_N, x_train_N) +
                                                               (sigma **2) * np.eye(len(x_train_N)))).dot(y_train_N)
    cov_arr = cov_func(x_grid_G, x_grid_G) - cov_func(x_grid_G, x_train_N).\
        dot(np.linalg.inv(cov_func(x_train_N, x_train_N)+ (sigma **2) * np.eye(len(x_train_N)))).\
        dot(cov_func(x_train_N, x_grid_G))

    prng = np.random.RandomState(int(random_seed))
    for _ in range(n_samples):
        posterior_samples.append(prng.multivariate_normal(new_mean, cov_arr))
    posterior_samples = np.array(posterior_samples).reshape(n_samples, len(x_grid_G))
    return posterior_samples


def plot_samples(x_grid_G, samples, tag ='prior', l_value = 0.25, nu_value = None, err=0):
    for i, sample in enumerate(samples):
        plt.plot(x_grid_G, sample, label='{}'.format(i+1), linestyle='-.')
        plt.legend(loc='best')
        if not nu_value:
            plt.title('{} using a square expnential kernel with L_value: {}'.format(tag, l_value))
            plt.savefig("./figs/{}/L = {}.png".format(tag, l_value))
        else:
            plt.title('{} using a matern kernel with L_value: {}, nu_value: {}'.format(tag, l_value, nu_value))
            plt.savefig("./figs/{}/L = {}_NU = {}.png".format(tag, l_value, nu_value))

def cov_func(X, Y):
    cov_value = matern_kernel_func(X, Y)
    return cov_value

def main():
    
    setup_env()
    x_grid_G = np.linspace(-20, 20, num=200)
    mean = np.zeros(len(x_grid_G))
    prior_samples = draw_GP_prior_samples_at_x_grid(x_grid_G, mean, cov_func, random_seed=42, n_samples=5)
    plot_samples(x_grid_G, prior_samples, tag='prior', nu_value=NU, l_value=L)
    x_train_N = np.asarray([-2., -1.8, -1., 1., 1.8, 2.])
    y_train_N = np.asarray([-3., 0.2224, 3., 3., 0.2224, -3.])
    posterior_samples = draw_GP_posterior_samples_at_x_grid(x_train_N, y_train_N, x_grid_G, mean, cov_func, sigma=0.1,
                                        random_seed=42, n_samples=5)

    plot_samples(x_grid_G, posterior_samples, tag='posterior', l_value=L, nu_value=NU, err=0.1)

if __name__== "__main__":
    main()