import os, sys

sys.path.append('//Users/minghao/Research/Projects/mylib/py3/')
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

import corner
# prepare final table for fesc and EW modeling

def add_EW_and_fesc(tbl, outdir):
    '''
    Given the output from bilby fitting, compute the EW and fesc for log-normal
    distribution fittings.

    Parameters
    -----

    '''

    EWmean = []
    EWsigma = []
    fescmean = []
    fescsigma = []

    if 'J0100' in outdir:
        field = 'J0100'
        prog_id = '4713'
    elif 'J1148' in outdir:
        field = 'J1148'
        prog_id = '3117'

    for index in range(len(tbl)):
        isource = prog_id+'_%d'%(tbl['NUMBER'][index])
        label = isource

        # read the fitting result
        result = bilby.result.read_in_result(outdir=outdir, label=label)
        samples = result.nested_samples
        weights = np.array(samples['weights'])
        flux_sample = dyfunc.resample_equal(np.array(samples['flux']), weights)
        A_sample = dyfunc.resample_equal(np.array(samples['A']), weights)

        z = tbl['z_O3doublet_combined_n'][index]
        EWsample = flux_sample / A_sample / tbl['z_O3doublet_combined_n'][index]

        # read info for fitting
        mag_F115W = tbl['MAG_AUTO_F115W_apcor'][index]
        mag_F200W = tbl['MAG_AUTO_F200W_apcor'][index]
        emag_F115W = tbl['enu_F115W_aper_model'][index]/\
                tbl['fnu_F115W_AUTO_apcor'][index]*2.5/np.log(10)
        emag_F200W = tbl['enu_F200W_aper_model'][index]/\
                tbl['fnu_F200W_AUTO_apcor'][index]*2.5/np.log(10)
        fHb = tbl['f_Hb'][index]
        fHb_err = tbl['f_Hb_err'][index]

        mags = [mag_F115W,mag_F200W]
        magerrs = [emag_F115W,emag_F200W]


        if tbl['fluxquantile'][index][0]>0:
            EWmean.append(np.median(EWsample))
            EWsigma.append(np.std(EWsample))
        elif tbl['fluxquantile'][index][-1]<0:
            EWmean.append(np.std(EWsample))
            EWsigma.append(-1)
        else:
            EWmean.append(tbl['EWquantile'][index][-1])
            EWsigma.append(-1)


        EW_from_Hb, eEW_from_Hb = LAEspec.EW_Lya_from_NIRCam(fHb, fHb_err, mags, magerrs, z)

        # we need to correct for extinction
        subtbl = tbl_prosp[(tbl_prosp['FIELD']==field)&(tbl_prosp['NUMBER']==tbl['NUMBER'][index])]
        if len(subtbl)>0:
            av = float(subtbl['EBV'][0]) * 3.43816749
            ahb = extinction.calzetti00(np.array([4863.0]), av, 3.43)
            hbcorr = 10**(0.4*ahb[0])
            #print(hbcorr)
        else:
            hbcorr = 1
        EW_from_Hb_corr = EW_from_Hb * hbcorr
        eEW_from_Hb_corr = eEW_from_Hb * hbcorr

        fescmean.append(EWmean[-1] / EW_from_Hb_corr)

        if fescmean[-1]<0:
            fescsigma.append(-1)
        else:
            fescsigma.append(fescmean[-1] * np.sqrt(np.std(EWsample)**2/np.median(EWsample)**2 + (eEW_from_Hb/EW_from_Hb)**2))

    tbl['EWmean'] = np.array(EWmean)
    tbl['EWsigma'] = np.array(EWsigma)
    tbl['fescmean'] = np.array(fescmean)
    tbl['fescsigma'] = np.array(fescsigma)

    return tbl


def get_mcmc_sample_from_table_EW(tbl):
    nwalkers = 10
    ndim = 2

    pos = np.array([0.2, 2]) + 0.1 * np.random.randn(nwalkers, ndim)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, LAEew.logprob_parameters_from_obs_list_table,\
            args=[tbl], pool=pool
        )
        sampler.run_mcmc(pos, 3000, progress=True)

    return sampler

def get_mcmc_sample_from_table_fesc(tbl, nrun=4000):
    nwalkers = 10
    ndim = 2

    pos = np.array([1, -2]) * (1 + 0.3 * np.random.randn(nwalkers, ndim))

    tbl = tbl[(~np.isnan(tbl['fescmean']))&(~np.isnan(tbl['fescsigma']))&(~np.isinf(tbl['fescmean']))]
    tbl['fescmean'][tbl['fescmean']>1] = 1

    print(tbl['fescmean'], tbl['fescsigma'])

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, LAEew.logprob_parameters_from_obs_list_table,\
            args=[tbl, 'fesc'], pool=pool
        )
        sampler.run_mcmc(pos, nrun, progress=True)

    return sampler

def fraction_higher_than_limit_lognorm(param, lim):
    # s is sigma, logscale is mu in the paper
    s, logscale = param
    scale = np.exp(logscale)
    return 1-lognorm.cdf(lim, s, 0, scale)

def fraction_higher_than_limit_lognorm_sample(lim, flat_samples):
    frac = []
    for p in flat_samples:
        frac.append(fraction_higher_than_limit_lognorm(p, lim))

    return frac

def prepare_subsets_for_fitting(direct='./', exclude_civ=True):
    allinfo_J0100 = Table.read(direct+'final_sample_J0100.fits')
    allinfo_J1148 = Table.read(direct+'final_sample_J1148.fits')

    z_J0100 = 6.327
    z_J1148 = 6.42

    dz_J0100 = (1+z_J0100)*2500/3e5
    dz_J1148 = (1+z_J1148)*2500/3e5

    if exclude_civ:
        allinfo_J0100 = allinfo_J0100[~allinfo_J0100['AGN']]
        allinfo_J1148 = allinfo_J1148[~allinfo_J1148['AGN']]
        print(len(allinfo_J0100), len(allinfo_J1148))

    allinfo_J0100_zQ = allinfo_J0100[\
                                (allinfo_J0100['zsys']>(z_J0100-dz_J0100))&\
                                (allinfo_J0100['zsys']<(z_J0100+dz_J0100))]
    allinfo_J1148_zQ = allinfo_J1148[\
                                (allinfo_J1148['zsys']>(z_J1148-dz_J1148))&\
                                (allinfo_J1148['zsys']<(z_J1148+dz_J1148))]
    allinfo_zQ = vstack([allinfo_J0100_zQ, allinfo_J1148_zQ])

    allinfo_J0100_zlow = allinfo_J0100[\
                                (allinfo_J0100['zsys']<(z_J0100-dz_J0100))&\
                                (allinfo_J0100['zsys']>6)]
    allinfo_J1148_zlow = allinfo_J1148[\
                                (allinfo_J1148['zsys']>6)&\
                                (allinfo_J1148['zsys']<(z_J1148-dz_J1148))]
    allinfo_zlow = vstack([allinfo_J0100_zlow, allinfo_J1148_zlow])

    allinfo_J0100_zhi = allinfo_J0100[\
                                (allinfo_J0100['zsys']>(z_J0100+dz_J0100))&\
                                (allinfo_J0100['zsys']<7)]
    allinfo_J1148_zhi = allinfo_J1148[\
                                (allinfo_J1148['zsys']<7)&\
                                (allinfo_J1148['zsys']>(z_J1148+dz_J1148))]
    allinfo_zhi = vstack([allinfo_J0100_zhi, allinfo_J1148_zhi])

    allinfo_fgbg = vstack([allinfo_zlow, allinfo_zhi])

    return allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ,\
            allinfo_zlow, allinfo_zhi, allinfo_fgbg

def prepare_subsets_for_fitting_new():
    dummy = 1


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
                prepare_subsets_for_fitting(exclude_civ=False)

    print(allinfo_J0100_zQ)
    print(allinfo_J1148_zQ)

    tbl_lit = Table.read('../data/kageura25.csv')
    tbl_lit.rename_column('zsys', 'redshift')
    tbl_lit_Muv_z = tbl_lit[(tbl_lit['redshift']>6)&(tbl_lit['redshift']<7)\
                            &(tbl_lit['MUV']>-24)&(tbl_lit['MUV']<-19)]

    print(len(allinfo_zQ), len(allinfo_zlow), len(allinfo_zhi))

    sample_zQ = get_mcmc_sample_from_table_EW(allinfo_zQ)
    pickle.dump(sample_zQ, open('../data/LAEfrac_saves/sample_EW_zQ_withCIV.sav', 'wb'))

    sample_zlow = get_mcmc_sample_from_table_EW(allinfo_zlow)
    pickle.dump(sample_zlow, open('../data/LAEfrac_saves/sample_EW_zlow_withCIV.sav', 'wb'))

    sample_zhi = get_mcmc_sample_from_table_EW(allinfo_zhi)
    pickle.dump(sample_zhi, open('../data/LAEfrac_saves/sample_EW_zhi_withCIV.sav', 'wb'))

    sample_J0100_zQ = get_mcmc_sample_from_table_EW(allinfo_J0100_zQ)
    pickle.dump(sample_J0100_zQ, open('../data/LAEfrac_saves/sample_EW_J0100_zQ_withCIV.sav', 'wb'))

    sample_J1148_zQ = get_mcmc_sample_from_table_EW(allinfo_J1148_zQ)
    pickle.dump(sample_J1148_zQ, open('../data/LAEfrac_saves/sample_EW_J1148_zQ_withCIV.sav', 'wb'))

    sample_fgbg = get_mcmc_sample_from_table_EW(allinfo_fgbg)
    pickle.dump(sample_fgbg, open('../data/LAEfrac_saves/sample_EW_fgbg_withCIV.sav', 'wb'))

    sample_lit = get_mcmc_sample_from_table_EW(tbl_lit_Muv_z)
    pickle.dump(sample_lit, open('../data/LAEfrac_saves/sample_EW_lit_withCIV.sav', 'wb'))

def test_evolution(factir):
    print('start')
    allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ,\
            allinfo_zlow, allinfo_zhi, allinfo_fgbg = \
                prepare_subsets_for_fitting(exclude_civ=True)

    allinfo_zQ['EWmean'] = allinfo_zQ['EWmean'] * factor
    allinfo_zQ['EWsigma'] = allinfo_zQ['EWsigma'] * factor

    sample_zQ = get_mcmc_sample_from_table_EW(allinfo_zQ)
    pickle.dump(sample_zQ, open(f'../data/LAEfrac_saves/sample_EW_zQ_factor{factor}.sav', 'wb'))



from scipy.special import erf

def get_mean_fesc(param):

    s, logscale = param
    scale = np.exp(logscale)
    A = 2/(1+erf(-logscale/np.sqrt(2)/s))

    grid = np.arange(0.001, 1, 0.001)
    weight = lognorm.pdf(grid, s=s, loc=0, scale=scale)

    mean = np.sum(grid * weight) / np.sum(weight)
    return mean

def fit_fesc():
    allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ,\
            allinfo_zlow, allinfo_zhi, allinfo_fgbg = \
                prepare_subsets_for_fitting(exclude_civ=True)
    print(allinfo_zQ['fescmean', 'fescsigma'])

#    sample_fesc_zQ = get_mcmc_sample_from_table_fesc(allinfo_zQ)
#    pickle.dump(sample_fesc_zQ, open('../data/LAEfrac_saves/sample_fesc_zQ_noCIV.sav', 'wb'))
    sample_fesc_zQ = pickle.load(open('../data/LAEfrac_saves/sample_fesc_zQ_noCIV.sav', 'rb'))


    # get the mean 

    meanlist = []

    flatsample = sample_fesc_zQ.get_chain(discard=3500,flat=True)
    corner.corner(flatsample, show_titles=True)
    plt.show()

    for param in flatsample:
        s, logscale = param
        scale = np.exp(logscale)
        mean = get_mean_fesc(param)
        print(s, logscale, mean)

        meanlist.append(mean)

    print(np.median(meanlist), np.percentile(meanlist, [16,84])-np.median(meanlist))

if __name__=='__main__':
    a = 1
#    fit_fesc()
#    fit_EW()
    for factor in np.arange(0.5,1, 0.1):
        test_evolution(factor)

