import os, sys

import LAE_EW_utils as LAEew
import LAE_spec_utils as LAEspec

import numpy as np
from astropy.io import fits
from scipy.optimize import minimize

import emcee
import pickle
import corner

import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack
from astropy import units

import bilby
from dynesty import utils as dyfunc
from mylib.spectrum.spec_measurement import Spectrum, read_filter, instru_filter_dict

class LineNotCoveredError(Exception):
    pass

import glob

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
plt.rcParams["lines.markeredgewidth"] = 2
plt.rcParams["patch.linewidth"] = 3
plt.rcParams["hatch.linewidth"] = 3
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


class Spec1dStacker:
    def __init__(self, speclist, zlist, restwavegrid=None):
        self.speclist = list(speclist)
        if restwavegrid is None:
            restwavegrid = np.arange(1150.17, 1300.17, 1)
        self.restwave = restwavegrid
        self.zlist = zlist

    def stack(self, weightlist=None):

        stackedspec = 0
        weightspec = 0
        varspec = 0

        if weightlist=='equal':
            print('equalweight')
            for spec in self.speclist:
                stackedspec += spec.restflux
                weightspec += np.ones(len(stackedspec))
                varspec += spec.restferr**2
        else:
            for spec in self.speclist:
                weightspec += spec.restferr**-2
                varspec += spec.restferr**-2
                stackedspec += spec.restflux * spec.restferr**-2

        stackedspec = stackedspec / weightspec
        stackederr = np.sqrt(varspec) / weightspec

        self.stackedspec = stackedspec
        self.stackederr = stackederr

    def mean_redshift(self):
        return np.mean(self.zlist)

def merge_spec_stacker(specstacker1, specstacker2):
    return Spec1dStacker(specstacker1.speclist + specstacker2.speclist,
                np.concatenate([specstacker1.zlist, specstacker2.zlist]))


def compile_stacked_spec(tbl_info, prog_id, specdir, savedir, Muvlim=-19.,\
                         zlim=[0,8], exclude_numbers=[], include_numbers=[],\
                        bootstrap=False):

    speclist = []
    restwavegrid = np.arange(1150.17, 1300.17, 1)

    zlist = []

    for index in range(len(tbl_info)):

        z = tbl_info['z_O3doublet_combined_n'][index]
        if z<zlim[0] or z>zlim[1]:
            continue

        isource = '%d_%d'%(prog_id, tbl_info['NUMBER'][index])
        label = isource

        if len(include_numbers)>0:
            if not tbl_info['NUMBER'][index] in include_numbers:
                continue

        else:
            if tbl_info['NUMBER'][index] in exclude_numbers:
                continue

            if not os.path.exists(savedir+f'/{isource}_result.json'):
                continue

        # find the spec1d file
        spec1dfile = os.path.join(specdir,'spec1d_%s_stack.fits'%isource)
        if not os.path.exists(spec1dfile):
            spec1dfile = os.path.join(specdir,'manu1d_%s_stack.fits'%isource)

        # force a detection limit?
        # let's restrict to 1-sigma detections
        lyaflux = tbl_info['fluxquantile'][index][1]
        if lyaflux < 0:
            continue

        if tbl_info['Muv'][index]>Muvlim:
            continue

        zlist.append(z)
        spec = LAEspec.PypeItSpecStacked(filename=spec1dfile, zO3=z,\
                                         outdir=savedir, label=label)
        spec.subtract_lya_cont(restwavegrid=restwavegrid)
        speclist.append(spec)

    if bootstrap:
        speclist = np.random.choice(speclist, size=len(speclist))

    stacker = Spec1dStacker(speclist, zlist=zlist, restwavegrid=restwavegrid)
    return stacker

def read_stacked_spec():
    # exclude objects that are: with strong CIV; contaminated by a nearby source; Lya not covered. 
    exclude_numbers_J0100 = [10287, 9970, 7230, 2159]
    exclude_numbers_J1148 = [3607, 9314, 11709, 9192]
    zq_J0100, zq_J1148 = 6.327, 6.42
    dz_J0100 = 2500/3e5*(1+zq_J0100)
    dz_J1148 = 2500/3e5*(1+zq_J1148)

    allinfo_J0100 = Table.read('./EW_fesc_J0100_msaexp_pypeit_beta_civ_av.fits')
    allinfo_J1148 = Table.read('./EW_fesc_J1148_msaexp_pypeit_beta_civ_av.fits')

    specdir_J0100 = '../../spectra/J0100_field/'
    savedir_J0100 = './lya_fitting/lya_saves_J0100/'

    specdir_J1148 = '../../spectra/J1148_field/'
    savedir_J1148 = './lya_fitting/lya_saves_J1148/'

    speclist_J0100_zQ = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100-dz_J0100,zq_J0100+dz_J0100],\
                        exclude_numbers=exclude_numbers_J0100)
    speclist_J0100_zlo = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[6,zq_J0100-dz_J0100],\
                        exclude_numbers=exclude_numbers_J0100)
    speclist_J0100_zhi = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100+dz_J0100,7],\
                        exclude_numbers=exclude_numbers_J0100)

    speclist_J1148_zQ = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148-dz_J1148,zq_J1148+dz_J1148],\
                        exclude_numbers=exclude_numbers_J1148)
    speclist_J1148_zlo = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[6,zq_J1148-dz_J1148],\
                        exclude_numbers=exclude_numbers_J1148)
    speclist_J1148_zhi = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148+dz_J1148,7],\
                        exclude_numbers=exclude_numbers_J1148)

    all_zlo = merge_spec_stacker(speclist_J1148_zlo, speclist_J0100_zlo)
    all_zhi = merge_spec_stacker(speclist_J1148_zhi, speclist_J0100_zhi)
    all_zQ = merge_spec_stacker(speclist_J1148_zQ, speclist_J0100_zQ)

    weighting = 'ivar'
    speclist_J0100_zQ.stack(weightlist=weighting)
    speclist_J1148_zQ.stack(weightlist=weighting)

    all_zlo.stack(weightlist=weighting)
    all_zhi.stack(weightlist=weighting)
    all_zQ.stack(weightlist=weighting)

    return speclist_J0100_zQ, speclist_J1148_zQ, all_zlo, all_zhi, all_zQ


def bootstrap_stacked_spec(n_bootstrap):
    # for J0100
    exclude_numbers_J0100 = [10287, 9970, 7230, 2159]
    exclude_numbers_J1148 = [3607, 9314, 11709, 9192]
    zq_J0100, zq_J1148 = 6.327, 6.42
    dz_J0100 = 2500/3e5*(1+zq_J0100)
    dz_J1148 = 2500/3e5*(1+zq_J1148)

    allinfo_J0100 = Table.read('./EW_fesc_J0100_msaexp_pypeit_beta_civ_v2.fits')
    allinfo_J1148 = Table.read('./EW_fesc_J1148_msaexp_pypeit_beta_civ_v2.fits')

    specdir_J0100 = '../../spectra/J0100_field/'
    savedir_J0100 = './lya_fitting/lya_saves_J0100/'

    specdir_J1148 = '../../spectra/J1148_field/'
    savedir_J1148 = './lya_fitting/lya_saves_J1148/'

    speclist_J0100_zQ_bootstrap = []
    speclist_J1148_zQ_bootstrap = []
    all_zlo_bootstrap = []
    all_zhi_bootstrap = []
    all_zQ_bootstrap = []

    for index in range(n_bootstrap):
        speclist_J0100_zQ = compile_stacked_spec(allinfo_J0100, 4713,\
                            specdir_J0100, savedir_J0100,\
                            zlim=[zq_J0100-dz_J0100,zq_J0100+dz_J0100],\
                            exclude_numbers=exclude_numbers_J0100,bootstrap=True)
        speclist_J0100_zlo = compile_stacked_spec(allinfo_J0100, 4713,\
                            specdir_J0100, savedir_J0100,\
                            zlim=[6,zq_J0100-dz_J0100],\
                            exclude_numbers=exclude_numbers_J0100,bootstrap=True)
        speclist_J0100_zhi = compile_stacked_spec(allinfo_J0100, 4713,\
                            specdir_J0100, savedir_J0100,\
                            zlim=[zq_J0100+dz_J0100,7],\
                            exclude_numbers=exclude_numbers_J0100,bootstrap=True)

        speclist_J1148_zQ = compile_stacked_spec(allinfo_J1148, 3117,\
                            specdir_J1148, savedir_J1148,\
                            zlim=[zq_J1148-dz_J1148,zq_J1148+dz_J1148],\
                            exclude_numbers=exclude_numbers_J1148,bootstrap=True)
        speclist_J1148_zlo = compile_stacked_spec(allinfo_J1148, 3117,\
                            specdir_J1148, savedir_J1148,\
                            zlim=[6,zq_J1148-dz_J1148],\
                            exclude_numbers=exclude_numbers_J1148,bootstrap=True)
        speclist_J1148_zhi = compile_stacked_spec(allinfo_J1148, 3117,\
                            specdir_J1148, savedir_J1148,\
                            zlim=[zq_J1148+dz_J1148,7],\
                            exclude_numbers=exclude_numbers_J1148,bootstrap=True)

        all_zlo = merge_spec_stacker(speclist_J1148_zlo, speclist_J0100_zlo)
        all_zhi = merge_spec_stacker(speclist_J1148_zhi, speclist_J0100_zhi)
        all_zQ = merge_spec_stacker(speclist_J1148_zQ, speclist_J0100_zQ)

        weighting = 'ivar'
        speclist_J0100_zQ.stack(weightlist=weighting)
        speclist_J1148_zQ.stack(weightlist=weighting)

        all_zlo.stack(weightlist=weighting)
        all_zhi.stack(weightlist=weighting)
        all_zQ.stack(weightlist=weighting)

        speclist_J0100_zQ_bootstrap.append(speclist_J0100_zQ)
        speclist_J1148_zQ_bootstrap.append(speclist_J1148_zQ)
        all_zlo_bootstrap.append(all_zlo)
        all_zhi_bootstrap.append(all_zhi)
        all_zQ_bootstrap.append(all_zQ)

    return speclist_J0100_zQ_bootstrap, speclist_J1148_zQ_bootstrap,\
            all_zlo_bootstrap, all_zhi_bootstrap, all_zQ_bootstrap


def read_stacked_spec_civ():
    # for J0100
    include_numbers_J0100 = [10287, 9970, 7230]
    include_numbers_J1148 = [3607, 9314, 11709]
    zq_J0100, zq_J1148 = 6.33, 6.42
    dz = 0.05

    allinfo_J0100 = Table.read('./EW_fesc_J0100_msaexp_pypeit_beta_civ_av.fits')
    allinfo_J1148 = Table.read('./EW_fesc_J1148_msaexp_pypeit_beta_civ_av.fits')

    specdir_J0100 = '../../spectra/J0100_field/'
    savedir_J0100 = './lya_fitting/lya_saves_J0100/'

    specdir_J1148 = '../../spectra/J1148_field/'
    savedir_J1148 = './lya_fitting/lya_saves_J1148/'

    speclist_J0100_zQ = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100-dz,zq_J0100+dz],\
                        include_numbers=include_numbers_J0100)
    speclist_J0100_zlo = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[6,zq_J0100-dz],\
                        include_numbers=include_numbers_J0100)
    speclist_J0100_zhi = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100+dz,7],\
                        include_numbers=include_numbers_J0100)

    speclist_J1148_zQ = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148-dz,zq_J1148+dz],\
                        include_numbers=include_numbers_J1148)
    speclist_J1148_zlo = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[6,zq_J1148-dz],\
                        include_numbers=include_numbers_J1148)
    speclist_J1148_zhi = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148+dz,7],\
                        include_numbers=include_numbers_J1148)
    civlist = [speclist_J1148_zlo, speclist_J0100_zlo, \
               speclist_J1148_zhi, speclist_J0100_zhi]
    all_civ_zlo = merge_spec_stacker(speclist_J1148_zlo, speclist_J0100_zlo)
    all_civ_zhi = merge_spec_stacker(speclist_J1148_zhi, speclist_J0100_zhi)
    all_civ_fgbg = merge_spec_stacker(all_civ_zhi, all_civ_zhi)

    all_civ_zlo.stack()
    all_civ_zhi.stack()
    all_civ_fgbg.stack()

    return all_civ_zlo, all_civ_zhi, all_civ_fgbg


def plot_stacked_spec():
    speclist_J0100_zQ, speclist_J1148_zQ, all_zlo, all_zhi, all_zQ = \
        read_stacked_spec()

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[12,4])

    velocity = (all_zQ.restwave - 1215.67) / 1215.67 * 3e5
    nspec = len(all_zQ.speclist)
    zmean = all_zQ.mean_redshift()
    axes[0].step(velocity, all_zQ.stackedspec, where='mid', color='k')
    axes[0].errorbar(velocity, all_zQ.stackedspec,\
                     yerr=all_zQ.stackederr, fmt='.', color='k')
    axes[0].set_xlim([-1600,1600])
    axes[0].plot([-1600,1600] ,[0,0], 'r--')
    axes[0].plot([0,0] ,[-0.1,0.3], 'r--')
#    axes[0].set_title('$\mathrm{|z-z_Q|<0.05}$', fontsize=18)
    axes[0].set_title('Near Quasars', fontsize=18)


    arrowx = -6/1215.67 * 3e5
    arrowy = -0.1 + 0.9*(0.3+0.1)
    darrowx = 0
    darrowy = -0.1 * (0.3+0.1)

    axes[0].text(0.05, 0.9, r'$N=%d, \langle z \rangle=%.2f$'%(nspec, zmean),
                 fontsize=12, transform=axes[0].transAxes)
    nspec = len(all_zlo.speclist)
    zmean = all_zlo.mean_redshift()

    axes[1].step(velocity, all_zlo.stackedspec, where='mid',\
              label='6.28<z<6.38', color='k')
    axes[1].errorbar(velocity, all_zlo.stackedspec,\
                     yerr=all_zlo.stackederr, fmt='.', color='k')
    axes[1].set_xlim([-1600,1600])
    axes[1].plot([-1600,1600] ,[0,0], 'r--')
    axes[1].plot([0,0] ,[-0.1,0.25], 'r--')
    axes[1].set_title('Foreground', fontsize=18)
    axes[1].text(0.05, 0.9, r'$N=%d, \langle z \rangle=%.2f$'%(nspec, zmean),
                 fontsize=12, transform=axes[1].transAxes)

    nspec = len(all_zhi.speclist)
    zmean = all_zhi.mean_redshift()

    axes[2].step(velocity, all_zhi.stackedspec, where='mid',\
              label='6.28<z<6.38', color='k')
    axes[2].errorbar(velocity, all_zhi.stackedspec,\
                     yerr=all_zhi.stackederr, fmt='.', color='k')
    axes[2].set_xlim([-1600,1600])
    axes[2].set_title('$\mathrm{z_Q+0.05<z<7}$', fontsize=18)
    axes[2].plot([-1600,1600] ,[0,0], 'r--')
    axes[2].plot([0,0] ,[-0.06,0.18], 'r--')
    axes[2].set_title('Background', fontsize=18)
    axes[2].text(0.05, 0.9, r'$N=%d, \langle z \rangle=%.2f$'%(nspec, zmean),
                 fontsize=12, transform=axes[2].transAxes)


    plt.xlim(-1600,1600)
    for ax in axes:
        ax.set_xlabel('Velocity $\mathrm{[km~s^{-1}]}$', fontsize=16)
    axes[0].set_ylabel(r'Stacked Flux $\mathrm{[10^{-18}~erg~s^{-1}cm^{-2}{\AA}^{-1}]}$', fontsize=16)

    plt.tight_layout()
    plt.show()


def stack_civ_emitters():

    all_civ_zlo, all_civ_zhi, all_civ_fgbg = read_stacked_spec_civ()

#    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[12,4])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=[5,4])
    axes = [ax]

    velocity = (all_civ_fgbg.restwave - 1215.67) / 1215.67 * 3e5
    axes[0].step(velocity, all_civ_fgbg.stackedspec, where='mid',\
         label=r'Foreground + Backgrund', color='k')
    axes[0].errorbar(velocity, all_civ_fgbg.stackedspec,\
                yerr=all_civ_fgbg.stackederr, fmt='.', color='k')
    axes[0].set_xlim([-1600,1600])
    axes[0].plot([-1600,1600] ,[0,0], 'r--')
    axes[0].plot([0,0] ,[-0.1,0.4], 'r--')
    axes[0].set_title('Foreground and Background CIV Emitters', fontsize=16)

    nspec = len(all_civ_fgbg.speclist)
    zmean = all_civ_fgbg.mean_redshift()

    plt.xlim(-1600,1600)
    for ax in axes:
        ax.set_xlabel('Velocity $\mathrm{[km~s^{-1}]}$', fontsize=16)
    axes[0].set_ylabel(r'Stacked Flux $\mathrm{[10^{-18}~erg~s^{-1}cm^{-2}{\AA}^{-1}]}$', fontsize=16)
    axes[0].text(0.05, 0.9, r'$N=%d, \langle z \rangle=%.2f$'%(nspec, zmean),
                 fontsize=12, transform=axes[0].transAxes)

    plt.tight_layout()
    plt.savefig('stackedspec_for_civ.pdf')
    plt.show()

    plt.show()


def plot_bootstrap(n_bootstrap):
    speclist_J0100_zQ_bootstrap, speclist_J1148_zQ_bootstrap,\
            all_zlo_bootstrap, all_zhi_bootstrap, all_zQ_bootstrap =\
            bootstrap_stacked_spec(n_bootstrap)

    # get the spec list, for the GnQ first
    speclist = [spec.stackedspec for spec in all_zQ_bootstrap]
    meanspec = np.mean(speclist, axis=0)
    errspec = np.std(speclist, axis=0)

    velocity = (all_zQ_bootstrap[0].restwave - 1215.67) / 1215.67 * 3e5

    plt.errorbar(velocity, meanspec,\
                yerr=errspec, fmt='.', color='k')
    plt.step(velocity, meanspec, where='mid',\
         label=r'Foreground + Backgrund', color='k')

    plt.xlim([-1600,1600])
    plt.show()


    speclist = [spec.stackedspec for spec in all_zQ_bootstrap]
    meanspec = np.mean(speclist, axis=0)
    errspec = np.std(speclist, axis=0)

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[12,4])


    axes[0].step(velocity, meanspec, where='mid', color='k')
    axes[0].errorbar(velocity, meanspec,\
                     yerr=errspec, fmt='.', color='k')
    axes[0].set_xlim([-1600,1600])
    axes[0].plot([-1600,1600] ,[0,0], 'r--')
    axes[0].plot([0,0] ,[-0.1,0.3], 'r--')
    axes[0].set_title('Near Quasars', fontsize=18)


    speclist = [spec.stackedspec for spec in all_zlo_bootstrap]
    meanspec = np.mean(speclist, axis=0)
    errspec = np.std(speclist, axis=0)

    axes[1].step(velocity, meanspec, where='mid', color='k')
    axes[1].errorbar(velocity, meanspec,\
                     yerr=errspec, fmt='.', color='k')
    axes[1].set_xlim([-1600,1600])
    axes[1].plot([-1600,1600] ,[0,0], 'r--')
    axes[1].plot([0,0] ,[-0.1,0.3], 'r--')
    axes[1].set_title('Foreground', fontsize=18)


    speclist = [spec.stackedspec for spec in all_zhi_bootstrap]
    meanspec = np.mean(speclist, axis=0)
    errspec = np.std(speclist, axis=0)

    axes[2].step(velocity, meanspec, where='mid', color='k')
    axes[2].errorbar(velocity, meanspec,\
                     yerr=errspec, fmt='.', color='k')
    axes[2].set_xlim([-1600,1600])
    axes[2].plot([-1600,1600] ,[0,0], 'r--')
    axes[2].plot([0,0] ,[-0.1,0.3], 'r--')
    axes[2].set_title('Background', fontsize=18)


    plt.xlim(-1600,1600)
    for ax in axes:
        ax.set_xlabel('Velocity $\mathrm{[km~s^{-1}]}$', fontsize=16)
    axes[0].set_ylabel(r'Stacked Flux $\mathrm{[10^{-18}~erg~s^{-1}cm^{-2}{\AA}^{-1}]}$', fontsize=16)

    plt.tight_layout()
    plt.savefig('stackedspec_bootstrap.pdf')
    plt.show()


if __name__=='__main__':
    plot_stacked_spec()# figure 2
#    stack_civ_emitters()# figure A1

#    bootstrap_stacked_spec(n_bootstrap=10)
#    plot_bootstrap(100)
