# code to make Figure 3

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.stats import lognorm, norm
#sys.path.append('/Users/myue/Research/Projects/JWST/dependencies/msa_spec_utils/')
sys.path.append('//Users/minghao/Research/Projects/mylib/py3/')

import LAE_EW_utils as LAEew
import LAE_spec_utils as LAEspec
from fit_LAE_frac import prepare_subsets_for_fitting
import fit_LAE_frac

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.markeredgewidth"] = 1
plt.rcParams["patch.linewidth"] = 1
plt.rcParams["hatch.linewidth"] = 1
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.handletextpad"] = 1
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["font.family"] = "Serif"
plt.rcParams["mathtext.default"] = "regular"

plt.rcParams.update({
    "text.usetex": True,})

def plot_ax_redshift_EW(ax, tbl, Muv_min, Muv_max, uplim=False, legend=False):
    tbl_to_plot = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max) & \
                      (~tbl['AGN'])]
    tbl_to_plot_AGN = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max) & \
                      (tbl['AGN'])]

    # plot the 1-sigma errorbar
    EWmed = tbl_to_plot['EWlya']
    EWsigma_low = tbl_to_plot['EWlya_le']
    EWsigma_hi = tbl_to_plot['EWlya_ue']

    if not uplim:
        ax.errorbar(tbl_to_plot['zsys'], y=EWmed,\
                yerr=(EWsigma_low, EWsigma_hi), fmt='o', color='k',\
                    label=r'$\mathrm{EW(CIV)<12{\AA}}$')
    else:
        ax.errorbar(tbl_to_plot['zsys'], EWmed,\
                    yerr=[5]*len(tbl_to_plot), fmt=r'_',\
                color='k', alpha=0.7, uplims=True)

    # plot AGNs
    EWmed = tbl_to_plot_AGN['EWlya']
    EWsigma_low = tbl_to_plot_AGN['EWlya_le']
    EWsigma_hi = tbl_to_plot_AGN['EWlya_ue']

    if not uplim:
        ax.errorbar(tbl_to_plot_AGN['zsys'], y=EWmed,\
                    yerr=(EWsigma_low, EWsigma_hi), fmt='x', color='k',\
                    label=r'$\mathrm{EW(CIV)>12{\AA}}$')
    else:
        ax.errorbar(tbl_to_plot_AGN['zsys'], EWmed,\
                    yerr=[5]*len(tbl_to_plot_AGN), fmt=r'x',\
                color='k', alpha=0.7, uplims=True)


    if legend:
        ax.legend(frameon=True, handletextpad=0.2)

    ax.set_xlabel('Redshift')

def plot_redshift_EW(Muv_min, Muv_max):
    tbl_J0100_plot = Table.read('./final_sample_J0100.fits')
    tbl_J1148_plot = Table.read('./final_sample_J1148.fits')

    tbl_J0100_plot = tbl_J0100_plot[tbl_J0100_plot['AGN']==0]
    tbl_J1148_plot = tbl_J1148_plot[tbl_J1148_plot['AGN']==0]

    tbl_J0100_uplim = tbl_J0100_plot[tbl_J0100_plot['EWlya_ue']<0]
    tbl_J0100_det = tbl_J0100_plot[tbl_J0100_plot['EWlya_ue']>0]
    tbl_J1148_uplim = tbl_J1148_plot[tbl_J1148_plot['EWlya_ue']<0]
    tbl_J1148_det = tbl_J1148_plot[tbl_J1148_plot['EWlya_ue']>0]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[9,3.8])

    dv_J0100 = 2500#4 * 70 * 7.327
    dv_J1148 = 2500#4 * 70 * 7.42
    dz_J0100 = 7.327*dv_J0100/3e5
    dz_J1148 = 7.42*dv_J1148/3e5

    axes[0].axvspan(6.327-dz_J0100, 6.327+dz_J0100, color='r', alpha=0.2, ec='none')
    axes[0].plot([6.327, 6.327], [-10,250], 'r--')
    axes[1].axvspan(6.42-dz_J1148, 6.42+dz_J1148, color='r', alpha=0.2, ec='none')
    axes[1].plot([6.42, 6.42], [-10,250], 'r--')
    plot_ax_redshift_EW(axes[0], tbl_J0100_det, Muv_min, Muv_max)
    plot_ax_redshift_EW(axes[1], tbl_J1148_det, Muv_min, Muv_max, legend=False)
    plot_ax_redshift_EW(axes[0], tbl_J0100_uplim, Muv_min, Muv_max, uplim=True)
    plot_ax_redshift_EW(axes[1], tbl_J1148_uplim, Muv_min, Muv_max,\
                        uplim=True, legend=False)

    axes[0].set_title('J0100 field', fontsize=16)
    axes[1].set_title('J1148 field', fontsize=16)
    axes[0].set_ylabel(r'Rest-frame Ly$\alpha$ EW [$\mathrm{\AA}$]')

    axes[0].plot([6, 7], [0, 0], 'k--')
    axes[1].plot([6, 7], [0, 0], 'k--')

    axes[0].set_xlim([6,7])
    axes[1].set_xlim([6,7])

    plt.tight_layout()
    plt.show()


def plot_ax_redshift_Llya(ax, tbl, Muv_min, Muv_max, uplim=False, legend=False):

    tbl_to_plot = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max) & \
                      (~tbl['AGN'])]
    tbl_to_plot_AGN = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max) & \
                      (tbl['AGN'])]

    lumdist = cosmo.luminosity_distance(tbl_to_plot['zsys']).to('cm').value
    lum_factor = lumdist ** 2 * 4*np.pi / 1e18 / 1e42

    # plot the 1-sigma errorbar
    med = tbl_to_plot['Flya'] * lum_factor
    sigma_low = tbl_to_plot['Flya_le'] * lum_factor
    sigma_hi = tbl_to_plot['Flya_ue'] * lum_factor


    if not uplim:
        ax.errorbar(tbl_to_plot['zsys'], y=med,\
                yerr=(sigma_low, sigma_hi), fmt='o', color='k',\
                    label=r'$\mathrm{EW(CIV)<12{\AA}}$')
    else:
        ax.errorbar(tbl_to_plot['zsys'], med,\
                    yerr=[1]*len(tbl_to_plot), fmt=r'_',\
                color='k', alpha=0.7, uplims=True)

    # plot AGNs
    EWmed = tbl_to_plot_AGN['EWlya']
    EWsigma_low = tbl_to_plot_AGN['EWlya_le']
    EWsigma_hi = tbl_to_plot_AGN['EWlya_ue']

    if not uplim:
        ax.errorbar(tbl_to_plot_AGN['zsys'], y=EWmed,\
                    yerr=(EWsigma_low, EWsigma_hi), fmt='x', color='k',\
                    label=r'$\mathrm{EW(CIV)>12{\AA}}$')
    else:
        ax.errorbar(tbl_to_plot_AGN['zsys'], EWmed,\
                    yerr=[5]*len(tbl_to_plot_AGN), fmt=r'x',\
                color='k', alpha=0.7, uplims=True)


    if legend:
        ax.legend(frameon=True, handletextpad=0.2)

    ax.set_xlabel('Redshift')


def plot_redshift_Llya(Muv_min, Muv_max):
    tbl_J0100_plot = Table.read('./final_sample_J0100.fits')
    tbl_J1148_plot = Table.read('./final_sample_J1148.fits')

    tbl_J0100_plot = tbl_J0100_plot[tbl_J0100_plot['AGN']==0]
    tbl_J1148_plot = tbl_J1148_plot[tbl_J1148_plot['AGN']==0]

    tbl_J0100_uplim = tbl_J0100_plot[tbl_J0100_plot['EWlya_ue']<0]
    tbl_J0100_det = tbl_J0100_plot[tbl_J0100_plot['EWlya_ue']>0]
    tbl_J1148_uplim = tbl_J1148_plot[tbl_J1148_plot['EWlya_ue']<0]
    tbl_J1148_det = tbl_J1148_plot[tbl_J1148_plot['EWlya_ue']>0]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[9,3.8])

    dv_J0100 = 2500#4 * 70 * 7.327
    dv_J1148 = 2500#4 * 70 * 7.42
    dz_J0100 = 7.327*dv_J0100/3e5
    dz_J1148 = 7.42*dv_J1148/3e5

    axes[0].axvspan(6.327-dz_J0100, 6.327+dz_J0100, color='r', alpha=0.2, ec='none')
    axes[0].plot([6.327, 6.327], [-2,10], 'r--')
    axes[1].axvspan(6.42-dz_J1148, 6.42+dz_J1148, color='r', alpha=0.2, ec='none')
    axes[1].plot([6.42, 6.42], [-2,10], 'r--')
    plot_ax_redshift_Llya(axes[0], tbl_J0100_det, Muv_min, Muv_max)
    plot_ax_redshift_Llya(axes[1], tbl_J1148_det, Muv_min, Muv_max, legend=False)
    plot_ax_redshift_Llya(axes[0], tbl_J0100_uplim, Muv_min, Muv_max, uplim=True)
    plot_ax_redshift_Llya(axes[1], tbl_J1148_uplim, Muv_min, Muv_max,\
                        uplim=True, legend=False)

    axes[0].set_title('J0100 field', fontsize=16)
    axes[1].set_title('J1148 field', fontsize=16)
    axes[0].set_ylabel(r'Ly$\alpha$ Luminosity [$\mathrm{\times10^{42}erg~s^{-1}}$]')

    axes[0].plot([6, 7], [0, 0], 'k--')
    axes[1].plot([6, 7], [0, 0], 'k--')

    axes[0].set_xlim([6,7])
    axes[0].set_ylim([-1.,7])
    axes[1].set_xlim([6,7])
    axes[1].set_ylim([-1.,7])

    plt.tight_layout()
    plt.savefig('./Llyascatter.pdf')
    plt.show()


# the following functions are for plotting the LAE frac
def EWfrac(param, EWlim):
    s, logscale = param
    scale = np.exp(logscale)
    return 1 - lognorm.cdf(EWlim, s, 0, scale)

def EWfrac_from_MCMC(EWlim, flat_samples):
    LAEfrac = []
    for p in flat_samples:
        LAEfrac.append(EWfrac(p, EWlim))

    return LAEfrac

def plot_LAE_frac(EWlim, plot_qso=True, plot_fgbg=True):

    allinfo_J0100_zQ, allinfo_J1148_zQ, allinfo_zQ, \
    allinfo_zlow, allinfo_zhi, allinfo_fgbg \
    = prepare_subsets_for_fitting('./', exclude_civ=True)

    sample_J1148_zQ = pickle.load(\
            open('../data/LAEfrac_saves/sample_EW_J1148_zQ_withCIV.sav', 'rb'))
    sample_J0100_zQ = pickle.load(\
            open('../data/LAEfrac_saves/sample_EW_J0100_zQ_withCIV.sav', 'rb'))
    sample_zlow = pickle.load(\
                open('../data/LAEfrac_saves/sample_EW_zlow_withCIV.sav', 'rb'))
    sample_zhi = pickle.load(\
                open('../data/LAEfrac_saves/sample_EW_zhi_withCIV.sav', 'rb'))
    sample_lit = pickle.load(\
                open('../data/LAEfrac_saves/sample_EW_lit_withCIV.sav', 'rb'))
    sample_fgbg = pickle.load(\
                open('../data/LAEfrac_saves/sample_EW_fgbg_withCIV.sav', 'rb'))
    sample_EW_zQ = pickle.load(\
                open('../data/LAEfrac_saves/sample_EW_zQ_withCIV.sav', 'rb'))

    flatsample_J1148_zQ = sample_J1148_zQ.get_chain(discard=2500, flat=True)
    EWfrac_J1148_zQ = EWfrac_from_MCMC(EWlim, flatsample_J1148_zQ)

    flatsample_J0100_zQ = sample_J0100_zQ.get_chain(discard=2500, flat=True)
    EWfrac_J0100_zQ = EWfrac_from_MCMC(EWlim, flatsample_J0100_zQ)
    
    flatsample_zlo = sample_zlow.get_chain(discard=2500, flat=True)
    EWfrac_zlo = EWfrac_from_MCMC(EWlim, flatsample_zlo)
    
    flatsample_zhi = sample_zhi.get_chain(discard=2500, flat=True)
    EWfrac_zhi = EWfrac_from_MCMC(EWlim, flatsample_zhi)
    
    flatsample_lit = sample_lit.get_chain(discard=2500, flat=True)
    EWfrac_lit = EWfrac_from_MCMC(EWlim, flatsample_lit)

    flatsample_zQ = sample_EW_zQ.get_chain(discard=2500, flat=True)
    EWfrac_zQ = fit_LAE_frac.fraction_higher_than_limit_lognorm_sample(EWlim, flatsample_zQ)

    fig, ax = plt.subplots(figsize=[8,4])

    if plot_qso:
        percentiles_J1148_Q = np.percentile(EWfrac_J1148_zQ, [16,50,84])
        ax.errorbar([6.42], [percentiles_J1148_Q[1]], \
                    yerr=[[percentiles_J1148_Q[1]-percentiles_J1148_Q[0]],\
                          [percentiles_J1148_Q[2]-percentiles_J1148_Q[1]]],\
                    fmt='^', color='red', label=r'J1148 field, $|z_Q-z|<0.05$', mfc='none')
        percentiles_J0100_Q = np.percentile(EWfrac_J0100_zQ, [16,50,84])
        ax.errorbar([6.33], [percentiles_J0100_Q[1]], \
                    yerr=[[percentiles_J0100_Q[1]-percentiles_J0100_Q[0]],\
                          [percentiles_J0100_Q[2]-percentiles_J0100_Q[1]]],\
                    fmt='v', color='red', label='J0100 field, $|z_Q-z|<0.05$', mfc='none')
        percentiles_zQ = np.percentile(EWfrac_zQ, [16,50,84])
        zQ = np.mean(allinfo_zQ['zsys'])
        ax.errorbar([zQ], [percentiles_zQ[1]], \
                    yerr=[[percentiles_zQ[1]-percentiles_zQ[0]], [percentiles_zQ[2]-percentiles_zQ[1]]],\
                    fmt='*', color='red', label='Both fields combined, $|z_Q-z|<0.05$', ms=10)

    if plot_fgbg:
        zlow = np.mean(allinfo_zlow['zsys'])
        percentiles_zlo = np.percentile(EWfrac_zlo, [16,50,84])
        ax.errorbar([zlow], [percentiles_zlo[1]], \
                    yerr=[[percentiles_zlo[1]-percentiles_zlo[0]],\
                          [percentiles_zlo[2]-percentiles_zlo[1]]],\
                    fmt='o', color='k', mfc='none', label='Foreground/Background')

        zhi = np.mean(allinfo_zhi['zsys'])
        percentiles_zhi = np.percentile(EWfrac_zhi, [16,50,84])
        ax.errorbar([zhi], [percentiles_zhi[1]], \
                    yerr=[[percentiles_zhi[1]-percentiles_zhi[0]],\
                          [percentiles_zhi[2]-percentiles_zhi[1]]],\
                    fmt='o', color='k', mfc='none')

    # literature
    ax.plot([6,7], [np.mean(EWfrac_lit)]*2, 'k--', label=r'JWST $6<z<7$ '+'(Kageura+25)')
#    print(np.percentile(EWfrac_lit,84))
    ax.fill_between([6,7], [np.percentile(EWfrac_lit,84)]*2, [np.percentile(EWfrac_lit,16)]*2, color='k',
                    alpha=0.2, edgecolor='None')

    ax.set_xlim([6,7])
    ax.set_ylim([0,0.8])

    ax.legend(fontsize=11)
    ax.set_xlabel('Redshift', fontsize=16)
    ax.set_ylabel(r'$\chi_\mathrm{LAE}(\mathrm{EW>%d\AA})$'%EWlim, fontsize=16)


    ax.tick_params(axis='both', direction='in')
    plotqso_suffix = '_qso' if plot_qso else ''
    plotfgbg_suffix = '_fgbg' if plot_fgbg else ''
#    plt.savefig(f'./Efrac_{EWlim}AA{plotfgbg_suffix}{plotqso_suffix}.pdf')

    flatsample_fgbg = sample_fgbg.get_chain(discard=2500, flat=True)

    plt.show()

if __name__=='__main__':


    # Figure 3

    plot_redshift_EW(Muv_min=-24, Muv_max=-19.)#top panel
    plot_redshift_Llya(Muv_min=-24, Muv_max=-19.)#middle panel
    plot_LAE_frac(EWlim, plot_qso=True, plot_fgbg=True)# bottom panel

