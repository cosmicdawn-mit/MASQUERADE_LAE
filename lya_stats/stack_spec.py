import os, sys
sys.path.append('/Users/myue/Research/Projects/JWST/dependencies/msa_spec_utils/')

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
    def __init__(self, speclist, restwavegrid=None):
        self.speclist = speclist
        if restwavegrid is None:
            restwavegrid = np.arange(1150.17, 1300.17, 1)
        self.restwave = restwavegrid

    def stack(self, weightlist=None):

        stackedspec = 0
        weightspec = 0

        if weightlist=='equal':
            for spec in self.speclist:
                stackedspec += spec.restflux
                weightspec += np.ones(len(stackedspec))
                print(stackedspec)
        else:
            for spec in self.speclist:
                stackedspec += spec.restflux * spec.restferr**-2
                weightspec += spec.restferr**-2

            stackedspec = stackedspec / weightspec

        self.stackedspec = stackedspec
        self.stackederr = weightspec**-0.5

def merge_spec_stacker(specstacker1, specstacker2):
    return Spec1dStacker(specstacker1.speclist + specstacker2.speclist)


def compile_stacked_spec(tbl_info, prog_id, specdir, savedir, Muvlim=-19.,\
                         zlim=[0,8], exclude_numbers=[], include_numbers=[]):

    speclist = []
    restwavegrid = np.arange(1150.17, 1300.17, 1)

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

        print(z, isource, tbl_info['Muv'][index])

        spec = LAEspec.PypeItSpecStacked(filename=spec1dfile, zO3=z,\
                                         outdir=savedir, label=label)
        spec.subtract_lya_cont(restwavegrid=restwavegrid)
        speclist.append(spec)

    stacker = Spec1dStacker(speclist, restwavegrid=restwavegrid)
    return stacker

def read_stacked_spec():
    # for J0100
    exclude_numbers_J0100 = [10287, 9970, 7230, 2159]
    exclude_numbers_J1148 = [3607, 9314, 11709, 14101, 9192]
    zq_J0100, zq_J1148 = 6.33, 6.42
    dz = 0.05

    allinfo_J0100 = Table.read('../data/lya_saves/lyainfo_J0100_msaexp_pypeit.fits')
    allinfo_J1148 = Table.read('../data/lya_saves/lyainfo_J1148_msaexp_pypeit.fits')

    specdir_J0100 = '../../ID4713/pypeit-reduc/data-release-2/reduced_spec_msaexp_pypeit/'
    savedir_J0100 = '../data/lya_saves/J0100/bilby_fit_lya_msaexp_v1/'

    specdir_J1148 = '../../ID3117/pypeit-reduc/data-release-2/reduced_spec_msaexp_pypeit/'
    savedir_J1148 = '../data/lya_saves/J1148/bilby_fit_lya_msaexp_v1/'

    speclist_J0100_zQ = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100-dz,zq_J0100+dz],\
                        exclude_numbers=exclude_numbers_J0100)
    speclist_J0100_zlo = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[6,zq_J0100-dz],\
                        exclude_numbers=exclude_numbers_J0100)
    speclist_J0100_zhi = compile_stacked_spec(allinfo_J0100, 4713,\
                        specdir_J0100, savedir_J0100,\
                        zlim=[zq_J0100+dz,7],\
                        exclude_numbers=exclude_numbers_J0100)

    speclist_J1148_zQ = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148-dz,zq_J1148+dz],\
                        exclude_numbers=exclude_numbers_J1148)
    speclist_J1148_zlo = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[6,zq_J1148-dz],\
                        exclude_numbers=exclude_numbers_J1148)
    speclist_J1148_zhi = compile_stacked_spec(allinfo_J1148, 3117,\
                        specdir_J1148, savedir_J1148,\
                        zlim=[zq_J1148+dz,7],\
                        exclude_numbers=exclude_numbers_J1148)

    all_zlo = merge_spec_stacker(speclist_J1148_zlo, speclist_J0100_zlo)
    all_zhi = merge_spec_stacker(speclist_J1148_zhi, speclist_J0100_zhi)
    all_zQ = merge_spec_stacker(speclist_J1148_zQ, speclist_J0100_zQ)

    speclist_J0100_zQ.stack()
    speclist_J1148_zQ.stack()

    all_zlo.stack()
    all_zhi.stack()
    all_zQ.stack()

    return speclist_J0100_zQ, speclist_J1148_zQ, all_zlo, all_zhi, all_zQ

def plot_stacked_spec():
    speclist_J0100_zQ, speclist_J1148_zQ, all_zlo, all_zhi, all_zQ = \
        read_stacked_spec()
    '''
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=[8,5])

    velocity = (speclist_J1148_zQ.restwave - 1215.67) / 1215.67 * 3e5
    axes[0][0].step(velocity, speclist_J0100_zQ.stackedspec, where='mid',\
                 label='6.28<z<6.38', color='k')
    axes[0][0].errorbar(velocity, speclist_J0100_zQ.stackedspec,\
                        yerr=speclist_J0100_zQ.stackederr, fmt='.', color='k')
    axes[0][0].set_xlim([-1600,1600])
    axes[0][0].plot([-1600,1600] ,[0,0], 'r--')
    axes[0][0].plot([0,0] ,[-0.1,0.25], 'r--')
    axes[0][0].set_title('J0100 field, $\mathrm{|z-z_Q|<0.05}$', fontsize=14)

    axes[0][1].step(velocity, speclist_J1148_zQ.stackedspec, where='mid',\
                 label='6.28<z<6.38', color='k')
    axes[0][1].errorbar(velocity, speclist_J1148_zQ.stackedspec,\
                        yerr=speclist_J1148_zQ.stackederr, fmt='.', color='k')
    axes[0][1].set_xlim([-1600,1600])
    axes[0][1].set_title('J1148 field, $\mathrm{|z-z_Q|<0.05}$', fontsize=14)
    axes[0][1].plot([-1600,1600] ,[0,0], 'r--')
    axes[0][1].plot([0,0] ,[-0.1,0.25], 'r--')

    axes[1][0].step(velocity, all_zlo.stackedspec, where='mid',\
                 label='6.28<z<6.38', color='k')
    axes[1][0].errorbar(velocity, all_zlo.stackedspec,\
                        yerr=all_zlo.stackederr, fmt='.', color='k')
    axes[1][0].set_xlim([-1600,1600])
    axes[1][0].set_title('All fields,  $\mathrm{6<z<z_Q-0.05}$', fontsize=14)
    axes[1][0].plot([-1600,1600] ,[0,0], 'r--')
    axes[1][0].plot([0,0] ,[-0.1,0.25], 'r--')

    axes[1][1].step(velocity, all_zhi.stackedspec, where='mid',\
                 label='6.28<z<6.38', color='k')
    axes[1][1].errorbar(velocity, all_zhi.stackedspec,\
                        yerr=all_zhi.stackederr, fmt='.', color='k')
    axes[1][1].set_xlim([-1600,1600])
    axes[1][1].set_title('All fields,  $\mathrm{z_Q+0.05<z<7}$', fontsize=14)
    axes[1][1].plot([-1600,1600] ,[0,0], 'r--')
    axes[1][1].plot([0,0] ,[-0.06,0.15], 'r--')

    '''

    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[12,4])

    velocity = (all_zQ.restwave - 1215.67) / 1215.67 * 3e5
    axes[0].step(velocity, all_zQ.stackedspec, where='mid',\
              label=r'$\mathrm{|z-z_Q|<0.05}$', color='k')
    axes[0].errorbar(velocity, all_zQ.stackedspec,\
                     yerr=all_zQ.stackederr, fmt='.', color='k')
    axes[0].set_xlim([-1600,1600])
    axes[0].plot([-1600,1600] ,[0,0], 'r--')
    axes[0].plot([0,0] ,[-0.1,0.3], 'r--')
    axes[0].set_title('$\mathrm{|z-z_Q|<0.05}$', fontsize=18)

    axes[1].step(velocity, all_zlo.stackedspec, where='mid',\
              label='6.28<z<6.38', color='k')
    axes[1].errorbar(velocity, all_zlo.stackedspec,\
                     yerr=all_zlo.stackederr, fmt='.', color='k')
    axes[1].set_xlim([-1600,1600])
    axes[1].set_title('$\mathrm{6<z<z_Q-0.05}$', fontsize=18)
    axes[1].plot([-1600,1600] ,[0,0], 'r--')
    axes[1].plot([0,0] ,[-0.1,0.25], 'r--')

    axes[2].step(velocity, all_zhi.stackedspec, where='mid',\
              label='6.28<z<6.38', color='k')
    axes[2].errorbar(velocity, all_zhi.stackedspec,\
                     yerr=all_zhi.stackederr, fmt='.', color='k')
    axes[2].set_xlim([-1600,1600])
    axes[2].set_title('$\mathrm{z_Q+0.05<z<7}$', fontsize=18)
    axes[2].plot([-1600,1600] ,[0,0], 'r--')
    axes[2].plot([0,0] ,[-0.06,0.15], 'r--')



    plt.xlim(-1600,1600)
    for ax in axes:
        ax.set_xlabel('Velocity $\mathrm{[km~s^{-1}]}$', fontsize=16)
    axes[0].set_ylabel(r'Stacked Flux $\mathrm{[10^{-18}~erg~s^{-1}cm^{-2}{\AA}^{-1}]}$', fontsize=16)

    plt.tight_layout()
    plt.savefig('stackedspec.pdf')
    plt.show()


if __name__=='__main__':
    plot_stacked_spec()
