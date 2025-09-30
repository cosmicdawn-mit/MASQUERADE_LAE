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


def logprob_parameters_from_obs_list_table_lngauss(params, tbl,\
                        mukey='EWmean', sigmakey='EWsigma'):
    # Tang+24b algorithm
    # https://academic.oup.com/mnras/article/531/2/2701/7681978

    # sum each row to get the total logprob

    # this is eq. 6
    logprob = 0
    for index in range(len(tbl)):
        redshift = tbl['redshift'][index]
        mu = tbl[mukey][index]
        sigma = tbl[sigmakey][index]

        if sigma>0:
            logprob += logprob_parameters_single_obj_gaussian(params, mu, sigma)
        else:
            logprob += logprob_parameters_single_obj_uplim(params, mu)

    # prior
    if params[0]<0:# or params[1]<0:
        return -np.inf

    return logprob


def logprob_parameters_from_obs_list_table_EW(params, tbl,\
                        mukey='EWmean', sigmakey='EWsigma'):
    # Tang+24b algorithm
    # https://academic.oup.com/mnras/article/531/2/2701/7681978

    #prior
    if params[0]<0 or params[1]<0 or params[0]>100 or params[1]>1000:
        return -np.inf

    # this is eq. 6
    logprob = 0
    for index in range(len(tbl)):
        redshift = tbl['redshift'][index]
        mu = tbl[mukey][index]
        sigma = tbl[sigmakey][index]

        uplim = sigma<0
        grid = np.arange(0.01, 200, 0.01)

        logprob += logprob_single_obj_lognorm(params, mu, sigma, grid, uplim=uplim)
#        else:
#            logprob += logprob_single_obj_uplim_exp(params, mu)


    return logprob


def logprob_parameters_from_obs_list_table_fesc(params, tbl):
    # Tang+24b algorithm
    # https://academic.oup.com/mnras/article/531/2/2701/7681978

    # sum each row to get the total logprob

    # this is eq. 6
    mukey='fescmean'
    sigmakey='fescsigma'

    # prior
    if params[0]<0.1 or params[1]<0 or params[0]>20 or params[1]>20:
        return -np.inf

    logprob = 0
    for index in range(len(tbl)):
        redshift = tbl['redshift'][index]
        mu = tbl[mukey][index]
        sigma = tbl[sigmakey][index]

        uplim = sigma < 0
#        if sigma>0:
        grid = np.arange(0.001, 1, 0.001)
        logprob += logprob_single_obj_lognorm(params, mu, sigma, grid, uplim=uplim)
        if np.isnan(logprob) or np.isinf(logprob):
            return -np.inf
    return logprob


def logprob_single_obj_lognorm(params, mu, sigma, grid, uplim=False):
    s, scale = params
    #scale = np.exp(logscale)

    if uplim:
        cdf = lognorm.cdf(mu, s, 0, scale)
        A = 2/(1+erf(-logscale/np.sqrt(2)/s))

        limprob = cdf
        return np.log(limprob)

    posterior_from_model = lognorm.pdf(grid, s, 0, scale)
#    A = 2/(1+erf(-logscale/np.sqrt(2)/s))
    A = 1
    posterior_from_obs = norm.pdf(grid, mu, sigma)

    # integrate
    intprob = np.trapz(posterior_from_model*posterior_from_obs, grid)*A
    if np.isnan(intprob) or np.isinf(intprob):
        return -np.inf
    return np.log(intprob)

def logprob_single_obj_uplim_exp(params, lim):
    # EW grid to be integrated

#    s, scale = params
#    limprob = lognorm.cdf(lim, s, 0, scale)
    limprob = 1-np.exp(-lim/params[0])

    return np.log(limprob)

def logprob_single_obj_uplim_truncnorm(params, lim, a_trunc=0, b_trunc=1):
    # EW grid to be integrated
    grid = np.linspace(a_trunc, b_trunc, 10000)
    loc, scale = params
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    limprob = truncnorm.cdf(grid, a=a, b=b, loc=loc, scale=scale)

    return np.log(limprob)


def logprob_single_obj_exp(params, mu, sigma):
    # EW grid to be integrated
    grid = np.arange(0.1, 1000, 0.1)

    EW0 = params[0]
    posterior_from_model = 1/EW0 * np.exp(-grid/EW0)
#    posterior_from_obs = norm.pdf(grid, mu, sigma)

    a_trunc, b_trunc = 0, 200
    a2, b2 = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
    posterior_from_obs = truncnorm.pdf(grid, a=a2, b=b2, loc=mu, scale=sigma)

    # integrate
    intprob = np.trapz(posterior_from_model*posterior_from_obs, grid)

    return np.log(intprob)


def logprob_single_obj_truncnorm(params, mu, sigma):
    # EW grid to be integrated
    grid = np.arange(0.01, 1, 0.01)

    loc, scale = params
    a_trunc, b_trunc = 0, 1
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    posterior_from_model = truncnorm.pdf(grid, a=a, b=b, loc=loc, scale=scale)

    a2, b2 = (a_trunc - mu) / sigma, (b_trunc - mu) / sigma
    posterior_from_obs = truncnorm.pdf(grid, a=a2, b=b2, loc=mu, scale=sigma)

#    posterior_from_model = posterior_from_model / np.trapz(posterior_from_model, grid)
#    posterior_from_obs = posterior_from_obs / np.trapz(posterior_from_obs, grid)
    # integrate
    intprob = np.trapz(posterior_from_model*posterior_from_obs, grid)

    return np.log(intprob)


