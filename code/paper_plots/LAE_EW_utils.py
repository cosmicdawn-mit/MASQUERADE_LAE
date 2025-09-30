import os, sys
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.special import erf
from scipy.stats import lognorm, norm, truncnorm
import bilby

from dynesty import utils as dyfunc
np.int = int

os.environ["OMP_NUM_THREADS"] = "1"

def logprob_parameters_from_obs_list_table(params, tbl, key='EW'):
    # Tang+24b algorithm
    # https://academic.oup.com/mnras/article/531/2/2701/7681978

    # make different setups for EW and fesc fitting
    if key=='EW':
        grid = np.arange(0.01, 200, 0.01)
        bound=False
        lims = [0, 5, -3, 5]
    elif key=='fesc':
        grid = np.arange(0.001, 1, 0.001)
        bound=True
        lims = [0, 5, -5, 0]

    else:
        raise KeyError('Unknown key for the table: %s'%key)

    #prior
    if params[0]<lims[0] or params[0]>lims[1]\
            or params[1]<lims[2] or params[1]>lims[3]:
        return -np.inf


    # this is eq. 6
    logprob = 0
    for index in range(len(tbl)):
#        redshift = tbl['redshift'][index]
        mu = tbl[key+'mean'][index]
        sigma = tbl[key+'sigma'][index]

        uplim = sigma<0

        logprob += logprob_single_obj_lognorm(params, mu, sigma, grid,\
                                              uplim=uplim, bound=bound)

    return logprob

# below are some probability distributions I tried.

def logprob_single_obj_lognorm(params, mu, sigma, grid, uplim=False,
                               bound=False):
    '''
    bound=None means no upper bound for the lognormal distribution.
    otherwise, the 
    '''

    s, logscale = params
    scale = np.exp(logscale)

    # according to the scipy document, log(x) should have
    # a standard deviation of s 
    # and a mean of log(scale)
    # in other words, s=sigma, logscale=mu in my paper

    # determine the normalization, if there is an upper limit
    if bound is False:
        A = 1
    else:
        A = 2/(1+erf(-logscale/np.sqrt(2)/s))

    if uplim:
        cdf = lognorm.cdf(mu, s, 0, scale) * A

        limprob = cdf
        return np.log(limprob)

    posterior_from_model = lognorm.pdf(grid, s, 0, scale)

    if bound is False:
        posterior_from_obs = norm.pdf(grid, mu, sigma)
    else:
        a_trunc, b_trunc = (0,1)
        a2, b2 = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
        posterior_from_obs = truncnorm.pdf(grid, a=a2, b=b2, loc=mu, scale=sigma)

    # integrate
    intprob = np.trapz(posterior_from_model*posterior_from_obs, grid)*A
    if np.isnan(intprob) or np.isinf(intprob):
        return -np.inf
    return np.log(intprob)

def logprob_single_obj_exp(params, mu, sigma, grid, uplim=False, bound=False):
    # EW grid to be integrated

    x0 = params[0]

    if bound is False:
        A = 1
    else:
        A = 1/(1-np.exp(-1/x0))

    if uplim:
        limprob = (1-np.exp(-mu/params[0]))*A
        return np.log(limprob)

    posterior_from_model = 1/x0 * np.exp(-grid/x0) * A

    if bound is False:
        posterior_from_obs = norm.pdf(grid, mu, sigma)
    else:
        a_trunc, b_trunc = (0,1)
        a2, b2 = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
        posterior_from_obs = truncnorm.pdf(grid, a=a2, b=b2, loc=mu, scale=sigma)

    # integrate
    intprob = np.trapz(posterior_from_model*posterior_from_obs, grid)

    return np.log(intprob)

