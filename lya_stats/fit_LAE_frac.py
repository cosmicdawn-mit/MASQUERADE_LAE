import os, sys
sys.path.append('/Users/myue/Research/Projects/JWST/dependencies/msa_spec_utils/')

import LAE_EW_utils as LAEew
import LAE_spec_utils as LAEspec


import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, hstack, vstack
from scipy.stats import lognorm, norm
import bilby

import emcee
from dynesty import utils as dyfunc
np.int = int

import extinction
from multiprocessing import Pool
import pickle


def get_mcmc_sample_from_table_EW(tbl):
    nwalkers = 10
    ndim = 2

    pos = np.array([0.2, 2]) + 0.1 * np.random.randn(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, LAEew.logprob_parameters_from_obs_list_table_EW,\
            args=[tbl], pool=pool
        )
        sampler.run_mcmc(pos, 3000, progress=True)

    return sampler

def get_mcmc_sample_from_table_fesc(tbl, nrun=4000):
    nwalkers = 10
    ndim = 2

    pos = np.array([3, 1]) * (1 + 0.3 * np.random.randn(nwalkers, ndim))

    tbl = tbl[(~np.isnan(tbl['fescmean']))&(~np.isnan(tbl['fescsigma']))&(~np.isinf(tbl['fescmean']))]
    tbl['fescsigma'] = np.abs(tbl['fescsigma'])
    tbl['fescmean'][tbl['fescmean']>1] = 1

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, LAEew.logprob_parameters_from_obs_list_table_fesc,\
            args=[tbl], pool=pool
        )
        sampler.run_mcmc(pos, nrun, progress=True)

    return sampler

def EWfrac_single_param(param, EWlim):
    s, logscale = param
    scale = np.exp(logscale)
    return 1-lognorm.cdf(EWlim, s, 0, scale)

def EWfrac_sample(EWlim, flat_samples):
    LAEfrac = []
    for p in flat_samples:
        LAEfrac.append(EWfrac_single_param(p, EWlim))

    return LAEfrac

def fescfrac_single_param(param, lim):
    s, logscale = param
    scale = np.exp(logscale)
    #loc, scale = param
    #a_trunc, b_trunc = 0, 1
    #a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    #return 1 - truncnorm.cdf(lim, a=a, b=b, loc=loc, scale=scale)
    return 1-lognorm.cdf(lim, s, loc=0, scale=scale)

def fescfrac_sample(lim, flat_samples):
    LAEfrac = []
    for p in flat_samples:
        LAEfrac.append(fescfrac_single_param(p, lim))

    return LAEfrac

def prepare_subsets_for_fitting(direct='./'):
    allinfo_J0100 = Table.read(direct+'EW_fesc_J0100_msaexp_pypeit.fits')
    allinfo_J1148 = Table.read(direct+'EW_fesc_J1148_msaexp_pypeit.fits')

    allinfo_J1148['redshift'] = allinfo_J1148['z_O3doublet_combined_n']
    allinfo_J0100['redshift'] = allinfo_J0100['z_O3doublet_combined_n']

    z_J0100 = 6.33
    z_J1148 = 6.42

    CIV_IDs_J0100 = [7230, 9970, 10287]
    CIV_IDs_J1148 = [3607, 9314, 11709, 14101]

    bad_IDs_J0100 = [812, 452, 1228, 10287, 9970, 7230, 2159]
    bad_IDs_J1148 = [16089, 15411, 3607, 9314, 11709, 14101, 6406, 9192]

    allinfo_J0100_zQ = allinfo_J0100[(allinfo_J0100['z_O3doublet_combined_n']>(z_J0100-0.05))&\
                                (allinfo_J0100['z_O3doublet_combined_n']<(z_J0100+0.05))&\
                                (allinfo_J0100['Muv']<-19)&
                                (allinfo_J0100['Muv']>-24)&
                                (~np.isin(allinfo_J0100['NUMBER'], bad_IDs_J0100))]

    allinfo_J1148_zQ = allinfo_J1148[(allinfo_J1148['z_O3doublet_combined_n']>(z_J1148-0.05))&\
                                    (allinfo_J1148['z_O3doublet_combined_n']<(z_J1148+0.05))&\
                                    (allinfo_J1148['Muv']<-19)&
                                    (allinfo_J1148['Muv']>-24)&
                                    (~np.isin(allinfo_J1148['NUMBER'], bad_IDs_J1148))]

    allinfo_zQ = vstack([allinfo_J0100_zQ, allinfo_J1148_zQ])

    allinfo_J0100_zlow = allinfo_J0100[(allinfo_J0100['z_O3doublet_combined_n']>6)&\
                                (allinfo_J0100['z_O3doublet_combined_n']<(z_J0100-0.05))&\
                                (allinfo_J0100['Muv']<-19)&
                                (allinfo_J0100['Muv']>-24)&
                                (~np.isin(allinfo_J0100['NUMBER'], bad_IDs_J0100))]

    allinfo_J1148_zlow = allinfo_J1148[(allinfo_J1148['z_O3doublet_combined_n']>6)&\
                                    (allinfo_J1148['z_O3doublet_combined_n']<(z_J1148-0.05))&\
                                    (allinfo_J1148['Muv']<-19)&
                                    (allinfo_J1148['Muv']>-24)&
                                    (~np.isin(allinfo_J1148['NUMBER'], bad_IDs_J1148))]

    allinfo_zlow = vstack([allinfo_J0100_zlow, allinfo_J1148_zlow])

    allinfo_J0100_zhi = allinfo_J0100[(allinfo_J0100['z_O3doublet_combined_n']>(z_J0100+0.05))&\
                                (allinfo_J0100['z_O3doublet_combined_n']<7)&\
                                (allinfo_J0100['Muv']<-19)&
                                (allinfo_J0100['Muv']>-24)&
                                (~np.isin(allinfo_J0100['NUMBER'], bad_IDs_J0100))]

    allinfo_J1148_zhi = allinfo_J1148[(allinfo_J1148['z_O3doublet_combined_n']>(z_J0100+0.05))&\
                                    (allinfo_J1148['z_O3doublet_combined_n']<7)&\
                                    (allinfo_J1148['Muv']<-19)&
                                    (allinfo_J1148['Muv']>-24)&
                                    (~np.isin(allinfo_J1148['NUMBER'], bad_IDs_J1148))]

    allinfo_zhi = vstack([allinfo_J0100_zhi, allinfo_J1148_zhi])

    allinfo_fgbg = vstack([allinfo_zlow, allinfo_zhi])

    return allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ,\
            allinfo_zlow, allinfo_zhi, allinfo_fgbg


def plot_EWfrac(EWlim):
    flatsample_zQ = sample_zQ.get_chain(discard=2000, thin=20,flat=True)
    EWfrac_zQ = EWfrac_from_MCMC(EWlim, flatsample_zQ)

    flatsample_fgbg = sample_fgbg.get_chain(discard=2000, thin=20,flat=True)
    EWfrac_fgbg = EWfrac_from_MCMC(EWlim, flatsample_fgbg)

    flatsample_lit = sample_lit.get_chain(discard=2000, thin=20,flat=True)
    EWfrac_lit = EWfrac_from_MCMC(EWlim, flatsample_lit)

    fig, ax = plt.subplots(figsize=[6,4])

    percentiles_zQ = np.percentile(EWfrac_zQ, [16,50,84])

    ax.errorbar([zQ], [percentiles_zQ[1]], \
                yerr=[[percentiles_zQ[1]-percentiles_zQ[0]], [percentiles_zQ[2]-percentiles_zQ[1]]],\
                fmt='*', color='red', label='This work, $|z_Q-z|<0.05$', ms=10)

    zfgbg = np.mean(allinfo_fgbg['redshift'])
    percentiles_fgbg = np.percentile(EWfrac_fgbg, [16,50,84])
    ax.errorbar([zfgbg], [percentiles_fgbg[1]], \
                yerr=[[percentiles_fgbg[1]-percentiles_fgbg[0]], [percentiles_fgbg[2]-percentiles_fgbg[1]]],\
                fmt='*', color='k', label='This work, $|z_Q-z|>0.05$', ms=10)

    # literature
    z_kageura = np.mean(tbl_lit['redshift'])

    percentiles_k25 = np.percentile(EWfrac_lit, [16,50,84])
    ax.errorbar([z_kageura], y=[percentiles_k25[1]], 
                yerr=[[percentiles_k25[1]-percentiles_k25[0]], [percentiles_k25[2]-percentiles_k25[1]]],\
                fmt='s', color='grey', mfc='none', label='Kageura+25')

    # tang+24a,b
    z_tang = [5,6,7]
    meanfrac_tang = [0.3, 0.24, 0.15]
    uerr_tang = [0.07, 0.10, 0.09]
    lerr_tang = [0.07, 0.09, 0.08]
    ax.errorbar(z_tang, y=meanfrac_tang,
                yerr=[lerr_tang, uerr_tang],\
                fmt='D', color='grey', mfc='none', label='Tang+24b')

    ax.set_xlim([4.6,7.2])
    ax.legend(fontsize=12, loc=[0.02,0.02])
    ax.set_xlabel('Redshift', fontsize=16)
    ax.set_ylabel(r'$\chi_\mathrm{LAE}(\mathrm{EW>%d\AA})$'%EWlim, fontsize=16)


    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(f'./Efrac_{EWlim}AA.pdf')

    plt.show()

def fit_EW():
    allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ,\
            allinfo_zlow, allinfo_zhi, allinfo_fgbg = \
                prepare_subsets_for_fitting()

    tbl_lit = Table.read('../data/kageura25.csv')
    tbl_lit.rename_column('zsys', 'redshift')
    tbl_lit_Muv_z = tbl_lit[(tbl_lit['redshift']>6)&(tbl_lit['redshift']<7)\
                            &(tbl_lit['MUV']>-24)&(tbl_lit['MUV']<-19)]

    sample_zQ = get_mcmc_sample_from_table_EW(allinfo_zQ)
    pickle.dump(sample_zQ, open('../data/LAEfrac_saves/sample_EW_zQ_final0.sav', 'wb'))

    sample_zlow = get_mcmc_sample_from_table_EW(allinfo_zlow)
    pickle.dump(sample_zlow, open('../data/LAEfrac_saves/sample_EW_zlow_final0.sav', 'wb'))

    sample_zhi = get_mcmc_sample_from_table_EW(allinfo_zhi)
    pickle.dump(sample_zhi, open('../data/LAEfrac_saves/sample_EW_zhi_final0.sav', 'wb'))

    sample_J0100_zQ = get_mcmc_sample_from_table_EW(allinfo_J0100_zQ)
    pickle.dump(sample_J0100_zQ, open('../data/LAEfrac_saves/sample_EW_J0100_zQ_final0.sav', 'wb'))

    sample_J1148_zQ = get_mcmc_sample_from_table_EW(allinfo_J1148_zQ)
    pickle.dump(sample_J1148_zQ, open('../data/LAEfrac_saves/sample_EW_J1148_zQ_final0.sav', 'wb'))

    sample_fgbg = get_mcmc_sample_from_table_EW(allinfo_fgbg)
    pickle.dump(sample_fgbg, open('../data/LAEfrac_saves/sample_EW_fgbg_final0.sav', 'wb'))

    sample_lit = get_mcmc_sample_from_table_EW(tbl_lit_Muv_z)
    pickle.dump(sample_lit, open('../data/LAEfrac_saves/sample_EW_lit_final0.sav', 'wb'))

if __name__=='__main__':
#    test_fesc()
    fit_EW()

