import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.stats import lognorm, norm
sys.path.append('/Users/myue/Research/Projects/JWST/dependencies/msa_spec_utils/')

import LAE_EW_utils as LAEew
import LAE_spec_utils as LAEspec

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

def rescale_image(image, percentile=[5,95]):
    vmin, vmax = np.nanpercentile(image, percentile)
    rescaled_image = (image - vmin) / (vmax - vmin) * 255.0
    rescaled_image = np.clip(rescaled_image, 0, 255)
    return rescaled_image.astype(np.uint8)

def plot_field_J1148_talk():
    # read J0100 images
    J1148_F115W = fits.getdata('../data/NIRCam_images/J1148/stack_F115W_pipe4_v2_20230109.fits', 'SCI')
    J1148_F200W = fits.getdata('../data/NIRCam_images/J1148/stack_F200W_pipe4_v3_srcbkg_20230110.fits', 'SCI')
    J1148_F356W = fits.getdata('../data/NIRCam_images/J1148/stack_F356W_pipe4_v2_20230109.fits', 'SCI')


    J1148_blue = rescale_image(J1148_F115W, [0.5,99.5])
    J1148_green = rescale_image(J1148_F200W, [0.5,99.5])
    J1148_red = rescale_image(J1148_F356W, [0.5,99.5])

    J1148_rgb_image = np.stack((J1148_red, J1148_green, J1148_blue), axis=-1)

    # Plot the image
    hdu = fits.open('../data/NIRCam_images/J1148/stack_F115W_pipe4_v2_20230109.fits')[1]
    wcs = WCS(hdu.header)

    plt.subplot(projection=wcs)
    plt.imshow(J1148_rgb_image, origin='lower')
#    plt.title('J1148+5251 Field', fontsize=16)
    plt.axis('off')


    tbl_J1148 = Table.read('../../ID3117/pypeit-reduc/data-release-2/masterinfo_J1148_msaexp_pypeit.fits')
    tbl_J1148_lya = Table.read('../data/lya_saves/lyainfo_J1148_msaexp.fits')

    tbl_J1148 = tbl_J1148[tbl_J1148['redshift']>5.5]
    print(tbl_J1148)
    tbl_J1148_good = select_good_sources_to_plot(tbl_J1148_lya,\
                                                 'J1148', Muvlim=[-24,-1.])
    tbl_J1148_good = tbl_J1148_good[(tbl_J1148_good['EWquantile'][:,3]>10)&\
                                    (tbl_J1148_good['redshift']<6.47)&\
                                    (tbl_J1148_good['redshift']>6.37)]
    print(tbl_J1148_good)
    numbers = [int(number.split('_')[-1]) for number in tbl_J1148['isource']]
    tbl_J1148_good_plot = tbl_J1148[np.isin(numbers, tbl_J1148_good['NUMBER'])]

    ax = plt.gca()
    edgecolors_values = (tbl_J1148_good_plot['redshift'] - np.min(tbl_J1148_good_plot['redshift']))/\
                    (np.max(tbl_J1148_good_plot['redshift'])-np.min(tbl_J1148_good_plot['redshift']))
#    sc = ax.scatter(tbl_J1148_good_plot['RA'], tbl_J1148_good_plot['Dec'], transform=ax.get_transform('world'),
#               marker='o', fc='none', ec=plt.cm.coolwarm(edgecolors_values), lw=1)

    ax.scatter(177.0693592, 52.8639720, transform=ax.get_transform('world'),
               marker='s', fc='none', ec='y', lw=1)
#    ax.text(7550,4050, 'QSO (z=6.42)', color='w')

    plt.tight_layout()
    plt.savefig('J1148_field_talk.pdf', dpi=1000)
    plt.show()


def plot_field_J0100():
    # read J0100 images
    J0100_F115W = fits.getdata('../data/NIRCam_images/stack_F115W_pipe4_v4_pipeup_1.8.2_pub0988_20221028.fits')
    J0100_F200W = fits.getdata('../data/NIRCam_images/J0100/stack_F200W_pipe4_v4.1_pipeup_1.8.2_pub0988_20221031.fits')
    J0100_F356W = fits.getdata('../data/NIRCam_images/J0100/stack_F356W_pipe4_v4.1_pipeup_1.8.2_pub0988_20221030.fits')

    J0100_blue = rescale_image(J0100_F115W, [0.5,99.5])
    J0100_green = rescale_image(J0100_F200W, [0.5,99.5])
    J0100_red = rescale_image(J0100_F356W, [0.5,99.5])

    J0100_rgb_image = np.stack((J0100_red, J0100_green, J0100_blue), axis=-1)

    # Plot the image
    hdu = fits.open('../data/NIRCam_images/stack_F115W_pipe4_v4_pipeup_1.8.2_pub0988_20221028.fits')[1]
    wcs = WCS(hdu.header)

    plt.subplot(projection=wcs)
    plt.imshow(J0100_rgb_image, origin='lower')
    plt.title('J0100+2802 Field', fontsize=16)
    plt.axis('off')

    tbl_J0100 = Table.read('../../ID4713/pypeit-reduc/data-release-2/masterinfo_J0100_pypeit_msaexp_update.fits')
    tbl_J0100_lya = Table.read('../data/lya_saves/lyainfo_J0100_msaexp.fits')

    tbl_J0100 = tbl_J0100[tbl_J0100['redshift']>5.5]
    print(tbl_J0100)
    tbl_J0100_good = select_good_sources_to_plot(tbl_J0100_lya,\
                                                 'J0100', Muvlim=[-24,-19.])
    numbers = [int(number.split('_')[-1]) for number in tbl_J0100['isource']]
    tbl_J0100_good_plot = tbl_J0100[np.isin(numbers, tbl_J0100_good['NUMBER'])]


    ax = plt.gca()
    edgecolors_values = (tbl_J0100_good_plot['redshift'] - np.min(tbl_J0100_good_plot['redshift']))/\
                    (np.max(tbl_J0100_good_plot['redshift'])-np.min(tbl_J0100_good_plot['redshift']))
    sc = ax.scatter(tbl_J0100_good_plot['RA'], tbl_J0100_good_plot['Dec'], transform=ax.get_transform('world'),
               marker='o', fc='none', ec=plt.cm.coolwarm(edgecolors_values), lw=1)

    import matplotlib as mpl
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=np.min(tbl_J0100_good_plot['redshift']),\
                                vmax=np.max(tbl_J0100_good_plot['redshift']))

    cax = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, orientation='horizontal', label='Redshift', pad=0.02, aspect=50)

    # make the rotated arrows
    pixscale = 0.03
    arrowlength = 20/pixscale
    ymax, xmax = np.array(J0100_F356W).shape
    matrix = np.array([[hdu.header['PC1_1'], hdu.header['PC1_2']], 
                        [hdu.header['PC2_1'], hdu.header['PC2_2']]])
    dx1, dy1 = np.linalg.inv(matrix).dot(np.array([arrowlength, 0]))
    dx2, dy2 = np.linalg.inv(matrix).dot(np.array([0, arrowlength]))

    arrowx0 = xmax-30/pixscale
    arrowy0 = 30/pixscale
    ax.arrow(arrowx0, arrowy0, dx1, dy1, color='w', lw=1, head_width=2/pixscale)
    ax.arrow(arrowx0, arrowy0, dx2, dy2, color='w', lw=1, head_width=2/pixscale)

    ax.text(arrowx0+dx1+100, arrowy0+dy1-100, 'E', color='w')
    ax.text(arrowx0+dx2-200, arrowy0+dy1-200, 'N', color='w')

    ax.plot([20/pixscale, 40/pixscale], [20/pixscale, 20/pixscale], 'w-')
    ax.text(20/pixscale, 7/pixscale, '20"', color='w')

    # the quasar
    ax.scatter(15.0542787, 28.0405070, transform=ax.get_transform('world'),
               marker='s', fc='none', ec='w', lw=1)

    print(ymax, xmax)
    plt.savefig('J0100_field.pdf', dpi=1000)
    plt.show()

def plot_field_J1148():
    # read J0100 images
    J1148_F115W = fits.getdata('../data/NIRCam_images/J1148/stack_F115W_pipe4_v2_20230109.fits', 'SCI')
    J1148_F200W = fits.getdata('../data/NIRCam_images/J1148/stack_F200W_pipe4_v3_srcbkg_20230110.fits', 'SCI')
    J1148_F356W = fits.getdata('../data/NIRCam_images/J1148/stack_F356W_pipe4_v2_20230109.fits', 'SCI')


    J1148_blue = rescale_image(J1148_F115W, [0.5,99.5])
    J1148_green = rescale_image(J1148_F200W, [0.5,99.5])
    J1148_red = rescale_image(J1148_F356W, [0.5,99.5])

    J1148_rgb_image = np.stack((J1148_red, J1148_green, J1148_blue), axis=-1)

    # Plot the image
    hdu = fits.open('../data/NIRCam_images/J1148/stack_F115W_pipe4_v2_20230109.fits')[1]
    wcs = WCS(hdu.header)

    plt.subplot(projection=wcs)
    plt.imshow(J1148_rgb_image, origin='lower')
    plt.title('J1148+5251 Field', fontsize=16)
    plt.axis('off')


    tbl_J1148 = Table.read('../../ID3117/pypeit-reduc/data-release-2/masterinfo_J1148_msaexp_pypeit.fits')
    tbl_J1148_lya = Table.read('../data/lya_saves/lyainfo_J1148_msaexp.fits')

    tbl_J1148 = tbl_J1148[tbl_J1148['redshift']>5.5]
    print(tbl_J1148)
    tbl_J1148_good = select_good_sources_to_plot(tbl_J1148_lya,\
                                                 'J1148', Muvlim=[-24,-19.])
    numbers = [int(number.split('_')[-1]) for number in tbl_J1148['isource']]
    tbl_J1148_good_plot = tbl_J1148[np.isin(numbers, tbl_J1148_good['NUMBER'])]

    ax = plt.gca()
    edgecolors_values = (tbl_J1148_good_plot['redshift'] - np.min(tbl_J1148_good_plot['redshift']))/\
                    (np.max(tbl_J1148_good_plot['redshift'])-np.min(tbl_J1148_good_plot['redshift']))
    sc = ax.scatter(tbl_J1148_good_plot['RA'], tbl_J1148_good_plot['Dec'], transform=ax.get_transform('world'),
               marker='o', fc='none', ec=plt.cm.coolwarm(edgecolors_values), lw=1)

    import matplotlib as mpl
    cmap = mpl.cm.coolwarm
    norm = mpl.colors.Normalize(vmin=np.min(tbl_J1148['redshift']),\
                                vmax=np.max(tbl_J1148['redshift']))

    cax = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=ax, orientation='horizontal', label='Redshift', pad=0.02, aspect=50)

    # make the rotated arrows
    pixscale = 0.03
    arrowlength = 20/pixscale
    ymax, xmax = np.array(J1148_F356W).shape
    matrix = np.array([[hdu.header['PC1_1'], hdu.header['PC1_2']], 
                        [hdu.header['PC2_1'], hdu.header['PC2_2']]])
    dx1, dy1 = np.linalg.inv(matrix).dot(np.array([arrowlength, 0]))
    dx2, dy2 = np.linalg.inv(matrix).dot(np.array([0, arrowlength]))

    arrowx0 = xmax-30/pixscale
    arrowy0 = 30/pixscale+200
    ax.arrow(arrowx0, arrowy0, dx1, dy1, color='w', lw=1, head_width=2/pixscale)
    ax.arrow(arrowx0, arrowy0, dx2, dy2, color='w', lw=1, head_width=2/pixscale)

    ax.text(arrowx0+dx1+100, arrowy0+dy1-100, 'E', color='w')
    ax.text(arrowx0+dx2-200, arrowy0+dy2-400, 'N', color='w')

    ax.plot([20/pixscale, 40/pixscale],\
            [20/pixscale+200, 20/pixscale+200], 'w-')
    ax.text(20/pixscale, 7/pixscale+200, '20"', color='w')

    print(ymax, xmax)
    ax.set_xlim([100, xmax-100])
    ax.set_ylim([200, ymax-300])

    ax.scatter(177.0693592, 52.8639698, transform=ax.get_transform('world'),
               marker='s', fc='none', ec='w', lw=1)

    plt.savefig('J1148_field.pdf', dpi=1000)
    plt.show()


def plot_spec_examples():
    tbl_J0100 = Table.read('../data/lya_saves/lyainfo_J0100_msaexp.fits')
    tbl_J1148 = Table.read('../data/lya_saves/lyainfo_J1148_msaexp.fits')

#    print(tbl_J0100['redshift'])
    zqso_J0100, zqso_J1148 = 6.327, 6.42
    tbl_J0100_zqso = tbl_J0100[(tbl_J0100['redshift']>zqso_J0100-0.1)\
                               &(tbl_J0100['redshift']<zqso_J0100+0.1)\
                               &(tbl_J0100['fluxquantile'][:,0]>0)\
                               &(tbl_J0100['EWquantile'][:,3]>10)]

    tbl_J1148_zqso = tbl_J1148[(tbl_J1148['redshift']>zqso_J1148-0.1)\
                               &(tbl_J1148['redshift']<zqso_J1148+0.1)\
                               &(tbl_J1148['fluxquantile'][:,0]>0)
                               &(tbl_J1148['EWquantile'][:,3]>10)]

    tbl_J0100_zqso['PROGID'] = 4713
    tbl_J1148_zqso['PROGID'] = 3117

    tbl_master = vstack([tbl_J0100_zqso, tbl_J1148_zqso])
    tbl_master = tbl_master[(tbl_master['NUMBER']!=6406)]
    print(tbl_master)

    # plot the spectra
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=[9,6])
    axes_flat = axes.flatten()
    for index in range(len(axes_flat)):
        print(index)
        objid = int(tbl_master['NUMBER'][index])
        progid = int(tbl_master['PROGID'][index])
        field = 'J0100' if progid==4713 else 'J1148'

        redshift = tbl_master['redshift'][index]
#        if objid==6406:
#            continue
        spec1dfile = f'../../ID{progid}/pypeit-reduc/data-release-2/reduced_spec_msaexp_pypeit/spec1d_{progid}_{objid}_stack.fits'
        print(spec1dfile)

        spectbl = Table.read(spec1dfile)

        ax =  axes_flat[index]

        mask = spectbl['mask']
        wave = spectbl['wave'][np.where(mask)]
        wavemask = (wave>8000)&(wave<11000)
        flux = spectbl['flux'][np.where(mask)]

        ymin = 1.2*np.min(flux[wavemask])
        ymax = 1.5*np.max(flux[wavemask])
        ax.step(wave, flux, where='mid', color='k')
        ax.set_xlim([8000,11000])
        ax.set_ylim([ymin, ymax])
        textpos = ymin + (ymax-ymin) * 0.6
        ax.text(8100, textpos,\
                '%s_%s\nz=%.2f'%(field, objid, redshift), fontsize=12)


        lya_wave = 1215.67 * (1+redshift)

        ax.plot([lya_wave, lya_wave],\
                [1.1*np.max(flux[wavemask]) , 1.3*np.max(flux[wavemask])],
                'r-')

    fig.supxlabel(r'Observed Wavelength [Angstrom]', fontsize=14)
    fig.supylabel(r'Flux [$\mathdefault{\mu}$Jy]', fontsize=14)

    plt.tight_layout()
    plt.savefig('LAE_examples.pdf')
    plt.show()
#    print(tbl_J0100_zqso)
#    print(tbl_J1148_zqso[['NUMBER', 'redshift', 'Muv','EWquantile', 'FWHMquantile', 'zlyaquantile', 'fluxquantile']])
#    print(tbl_J1148_zqso['fluxquantile'][0])



def plot_spec_examples_talk():
    tbl_J0100 = Table.read('../data/lya_saves/lyainfo_J0100_msaexp.fits')
    tbl_J1148 = Table.read('../data/lya_saves/lyainfo_J1148_msaexp.fits')

#    print(tbl_J0100['redshift'])
    zqso_J0100, zqso_J1148 = 6.327, 6.42
    tbl_J0100_zqso = tbl_J0100[(tbl_J0100['redshift']>zqso_J0100-0.05)\
                               &(tbl_J0100['redshift']<zqso_J0100+0.05)\
                               &(tbl_J0100['fluxquantile'][:,0]>0)\
                               &(tbl_J0100['EWquantile'][:,3]>10)]

    tbl_J1148_zqso = tbl_J1148[(tbl_J1148['redshift']>zqso_J1148-0.05)\
                               &(tbl_J1148['redshift']<zqso_J1148+0.05)\
                               &(tbl_J1148['fluxquantile'][:,0]>0)
                               &(tbl_J1148['EWquantile'][:,3]>10)]

    tbl_J0100_zqso['PROGID'] = 4713
    tbl_J1148_zqso['PROGID'] = 3117

#    tbl_master = vstack([tbl_J0100_zqso, tbl_J1148_zqso])
    tbl_master = tbl_J1148_zqso
    tbl_master = tbl_master[(tbl_master['NUMBER']!=6406)]
    print(tbl_master)

    # plot the spectra
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=[18,3])
    axes_flat = axes.flatten()
    for index in range(len(axes_flat)):
        print(index)
        objid = int(tbl_master['NUMBER'][index])
        progid = int(tbl_master['PROGID'][index])
        field = 'J0100' if progid==4713 else 'J1148'

        redshift = tbl_master['redshift'][index]
#        if objid==6406:
#            continue
        spec1dfile = f'../../ID{progid}/pypeit-reduc/data-release-2/reduced_spec_msaexp_pypeit/spec1d_{progid}_{objid}_stack.fits'
        print(spec1dfile)

        spectbl = Table.read(spec1dfile)

        ax =  axes_flat[index]

        mask = spectbl['mask']
        wave = spectbl['wave'][np.where(mask)]
        wavemask = (wave>8000)&(wave<11000)
        flux = spectbl['flux'][np.where(mask)]

        ymin = 1.2*np.min(flux[wavemask])
        ymax = 1.5*np.max(flux[wavemask])
        ax.step(wave, flux, where='mid', color='k')
        ax.set_xlim([8500,10000])
        ax.set_ylim([ymin, ymax])
        textpos = ymin + (ymax-ymin) * 0.6
        ax.text(8550, textpos,\
                '%s_%s\nz=%.2f'%(field, objid, redshift), fontsize=16)


        lya_wave = 1215.67 * (1+redshift)

        ax.plot([lya_wave, lya_wave],\
                [1.1*np.max(flux[wavemask]) , 1.3*np.max(flux[wavemask])],
                'r-')

    axes[1].set_xlabel(r'Observed Wavelength [Angstrom]', fontsize=14)
    axes[0].set_ylabel(r'Flux [$\mathdefault{\mu}$Jy]', fontsize=14)

    plt.tight_layout()
    plt.savefig('LAE_examples_talk.pdf')
    plt.show()
#    print(tbl_J0100_zqso)
#    print(tbl_J1148_zqso[['NUMBER', 'redshift', 'Muv','EWquantile', 'FWHMquantile', 'zlyaquantile', 'fluxquantile']])
#    print(tbl_J1148_zqso['fluxquantile'][0])



def select_good_sources_to_plot(tbl, field, Muvlim=[-24,-19.]):
    if field == 'J0100':
        noLya_isource = [812, 452, 1228]
        AGN_isource = [10287, 9970, 7230]
        problem_isource = [2159]# 2159 is severely contaminated by a nearby galaxy

        badid_all = noLya_isource + AGN_isource + problem_isource# + badphot_isource
        basisource_all = ['4713_%d'%i for i in badid_all]

    elif field == 'J1148':
        noLya_isource = [16089, 15411, ]
        AGN_isource = [3607, 9314, 11709, 14101]
        problem_isource = [6406, 9192]# 9192 is severely contaminated by a nearby galaxy

        badid_all = noLya_isource + AGN_isource + problem_isource# + badphot_isource

    tbl_plot = tbl[~np.isin(tbl['NUMBER'], badid_all)]
    if Muvlim is not None:
        tbl_plot = tbl_plot[(tbl_plot['Muv']>Muvlim[0])&\
                            (tbl_plot['Muv']<Muvlim[1])]

    tbl_plot = tbl_plot[tbl_plot['z_O3doublet_combined_n']>6]
    return tbl_plot

def plot_ax_redshift_EW(ax, tbl, Muv_min, Muv_max, uplim=False):
    tbl_to_plot = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max)]

    # plot the 1-sigma errorbar
    EWmed = tbl_to_plot['EWquantile'][:,3]
    EWsigma_low = tbl_to_plot['EWquantile'][:,3] - tbl_to_plot['EWquantile'][:,2]
    EWsigma_hi = tbl_to_plot['EWquantile'][:,4] - tbl_to_plot['EWquantile'][:,3]
    EWmed = tbl_to_plot['EWmean']
    #EWsigma_low = tbl_to_plot['EWsigma']
    #EWsigma_hi = tbl_to_plot['EWsigma']
    if not uplim:
        ax.errorbar(tbl_to_plot['z_O3doublet_combined_n'], y=EWmed,\
                yerr=(EWsigma_low, EWsigma_hi), fmt='o', color='k')
    else:
        EWlim = tbl_to_plot['EWquantile'][:,-1]
        EWlim = EWmed + 3 * EWsigma_hi
        EWlim[EWlim<0] = 0
        ax.scatter(tbl_to_plot['z_O3doublet_combined_n'], EWlim, marker=r'$\downarrow$',\
                color='k', facecolor='none', alpha=0.7)

    ax.set_xlabel('Redshift')
#    ax.set_ylabel(r'Rest-frame Ly$\alpha$ EW [$\mathrm{\AA}$]')

def plot_ax_redshift_fesc(ax, tbl, Muv_min, Muv_max, uplim=False):
    tbl_to_plot = tbl[(tbl['Muv']>Muv_min) & (tbl['Muv']<Muv_max)]

    # plot the 1-sigma errorbar
    fescmean = tbl_to_plot['fescmean']
    fescsigma = tbl_to_plot['fescsigma']

    badmask = (np.isnan(fescmean))|(np.isinf(fescmean))|np.isnan(fescsigma)

#    EWsigma_low = tbl_to_plot['EWquantile'][:,3] - tbl_to_plot['EWquantile'][:,2]
#    EWsigma_hi = tbl_to_plot['EWquantile'][:,4] - tbl_to_plot['EWquantile'][:,3]
    if not uplim:
        ax.errorbar(tbl_to_plot['z_O3doublet_combined_n'][~badmask],\
                    y=fescmean[~badmask], yerr=fescsigma[~badmask],\
                    fmt='o', color='k')
    else:
        ax.plot(tbl_to_plot['z_O3doublet_combined_n'][~badmask],\
                fescmean[~badmask], r'$\downarrow$', color='k', mfc='none', lw=1)

    ax.set_xlabel('Redshift')


def plot_redshift_EW(Muv_min, Muv_max):
    tbl_J0100 = Table.read('EW_fesc_J0100_msaexp_pypeit.fits')
    tbl_J1148 = Table.read('EW_fesc_J1148_msaexp_pypeit.fits')

    tbl_J0100_plot = select_good_sources_to_plot(tbl_J0100, 'J0100')
    tbl_J1148_plot = select_good_sources_to_plot(tbl_J1148, 'J1148')

#    tbl_J0100_uplim = tbl_J0100_plot[tbl_J0100_plot['EWquantile'][:,0]<0]
#    tbl_J0100_det = tbl_J0100_plot[tbl_J0100_plot['EWquantile'][:,0]>0]
#    tbl_J1148_uplim = tbl_J1148_plot[tbl_J1148_plot['EWquantile'][:,0]<0]
#    tbl_J1148_det = tbl_J1148_plot[tbl_J1148_plot['EWquantile'][:,0]>0]
    tbl_J0100_uplim = tbl_J0100_plot[(tbl_J0100_plot['EWmean']<\
                                     3*tbl_J0100_plot['EWsigma'])|\
                                     (tbl_J0100_plot['EWsigma']<0)]
    tbl_J0100_det = tbl_J0100_plot[(tbl_J0100_plot['EWmean']>\
                                     3*tbl_J0100_plot['EWsigma'])&\
                                     (tbl_J0100_plot['EWsigma']>0)]

    tbl_J1148_uplim = tbl_J1148_plot[(tbl_J1148_plot['EWmean']<\
                                     3*tbl_J1148_plot['EWsigma'])|\
                                     (tbl_J1148_plot['EWsigma']<0)]

    tbl_J1148_det = tbl_J1148_plot[(tbl_J1148_plot['EWmean']>\
                                     3*tbl_J1148_plot['EWsigma'])&\
                                     (tbl_J1148_plot['EWsigma']>0)]


    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[9,3.8])

    axes[0].axvspan(6.33-0.05, 6.33+0.05, color='r', alpha=0.2, ec='none')
    axes[0].plot([6.33, 6.33], [-10,150], 'r--')
    axes[1].axvspan(6.42-0.05, 6.42+0.05, color='r', alpha=0.2, ec='none')
    axes[1].plot([6.42, 6.42], [-10,150], 'r--')
    plot_ax_redshift_EW(axes[0], tbl_J0100_det, Muv_min, Muv_max)
    plot_ax_redshift_EW(axes[1], tbl_J1148_det, Muv_min, Muv_max)
    plot_ax_redshift_EW(axes[0], tbl_J0100_uplim, Muv_min, Muv_max, uplim=True)
    plot_ax_redshift_EW(axes[1], tbl_J1148_uplim, Muv_min, Muv_max, uplim=True)

    axes[0].set_title('J0100 field', fontsize=16)
    axes[1].set_title('J1148 field', fontsize=16)
    axes[0].set_ylabel(r'Rest-frame Ly$\alpha$ EW [$\mathrm{\AA}$]')

    axes[0].plot([6, 7], [0, 0], 'k--')
    axes[1].plot([6, 7], [0, 0], 'k--')

    axes[0].set_xlim([6,7])
    axes[0].set_ylim([-10,150])
    axes[1].set_xlim([6,7])
    axes[1].set_ylim([-10,150])

    print(len(tbl_J1148_plot))
    print(len(tbl_J0100_plot))

    # print the lya frac
    tbl_J0100_zQ_lya = tbl_J0100_det[np.abs(tbl_J0100_det['z_O3doublet_combined_n']-6.33)<0.05]
    tbl_J0100_zQ_nolya = tbl_J0100_uplim[np.abs(tbl_J0100_uplim['z_O3doublet_combined_n']-6.33)<0.05]
    tbl_J1148_zQ_lya = tbl_J1148_det[np.abs(tbl_J1148_det['z_O3doublet_combined_n']-6.42)<0.05]
    tbl_J1148_zQ_nolya = tbl_J1148_uplim[np.abs(tbl_J1148_uplim['z_O3doublet_combined_n']-6.42)<0.05]

    N_QSO_no_Lya = len(tbl_J0100_zQ_nolya)+len(tbl_J1148_zQ_nolya)
    N_QSO_Lya = len(tbl_J0100_zQ_lya)+len(tbl_J1148_zQ_lya)
    print(N_QSO_Lya, N_QSO_no_Lya)

    # print the lya frac
    tbl_J0100_fgbg_lya = tbl_J0100_det[np.abs(tbl_J0100_det['z_O3doublet_combined_n']-6.33)>0.05]
    tbl_J0100_fgbg_nolya = tbl_J0100_uplim[np.abs(tbl_J0100_uplim['z_O3doublet_combined_n']-6.33)>0.05]
    tbl_J1148_fgbg_lya = tbl_J1148_det[np.abs(tbl_J1148_det['z_O3doublet_combined_n']-6.42)>0.05]
    tbl_J1148_fgbg_nolya = tbl_J1148_uplim[np.abs(tbl_J1148_uplim['z_O3doublet_combined_n']-6.42)>0.05]

    N_fgbg_no_Lya = len(tbl_J0100_fgbg_nolya)+len(tbl_J1148_fgbg_nolya)
    N_fgbg_Lya = len(tbl_J0100_fgbg_lya)+len(tbl_J1148_fgbg_lya)
    print(N_fgbg_Lya, N_fgbg_no_Lya)


    # plot the Bayesian analysis result

    plt.tight_layout()
    plt.savefig('./EWscatter.pdf')
    plt.show()

def plot_redshift_fesc(Muv_min, Muv_max):
    tbl_J0100 = Table.read('EW_fesc_J0100_msaexp_pypeit.fits')
    tbl_J1148 = Table.read('EW_fesc_J1148_msaexp_pypeit.fits')

    tbl_J0100_plot = select_good_sources_to_plot(tbl_J0100, 'J0100')
    tbl_J1148_plot = select_good_sources_to_plot(tbl_J1148, 'J1148')

    tbl_J0100_uplim = tbl_J0100_plot[tbl_J0100_plot['fescsigma']<0]
    tbl_J0100_det = tbl_J0100_plot[tbl_J0100_plot['fescsigma']>0]
    tbl_J1148_uplim = tbl_J1148_plot[tbl_J1148_plot['fescsigma']<0]
    tbl_J1148_det = tbl_J1148_plot[tbl_J1148_plot['fescsigma']>0]

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=[9,3.8])

    axes[0].axvspan(6.33-0.05, 6.33+0.05, color='r', alpha=0.2, ec='none')
    axes[0].plot([6.33, 6.33], [0,1], 'r--')
    axes[1].axvspan(6.42-0.05, 6.42+0.05, color='r', alpha=0.2, ec='none')
    axes[1].plot([6.42, 6.42], [0,1], 'r--')

    plot_ax_redshift_fesc(axes[0], tbl_J0100_det, Muv_min, Muv_max)
    plot_ax_redshift_fesc(axes[1], tbl_J1148_det, Muv_min, Muv_max)
    plot_ax_redshift_fesc(axes[0], tbl_J0100_uplim, Muv_min, Muv_max, uplim=True)
    plot_ax_redshift_fesc(axes[1], tbl_J1148_uplim, Muv_min, Muv_max, uplim=True)

    axes[0].set_title('J0100 field', fontsize=16)
    axes[1].set_title('J1148 field', fontsize=16)
    axes[0].set_ylabel(r'$f_\mathrm{esc}^{\mathrm{Ly}\alpha}$')

    axes[0].plot([5.8, 7], [0, 0], 'k--')
    axes[1].plot([5.8, 7], [0, 0], 'k--')

    axes[0].set_xlim([5.8,7])
    axes[0].set_ylim([0,1])
    axes[1].set_xlim([5.8,7])
    axes[1].set_ylim([0,1])

    # plot the Bayesian analysis result

    plt.tight_layout()
    plt.savefig('./fescscatter.pdf')
    plt.show()

# the following functions are for plotting the LAE frac
def EWfrac(param, EWlim):
    s, scale = param
    return 1 - lognorm.cdf(EWlim, s, 0, scale)

def EWfrac_from_MCMC(EWlim, flat_samples):
    LAEfrac = []
    for p in flat_samples:
        LAEfrac.append(EWfrac(p, EWlim))

    return LAEfrac

def plot_LAE_frac(EWlim):

    sample_J1148_zQ = pickle.load(open('../data/LAEfrac_saves/sample_J1148_zQ_msaexp_pypeit.sav', 'rb'))
    sample_J0100_zQ = pickle.load(open('../data/LAEfrac_saves/sample_J0100_zQ_msaexp_pypeit.sav', 'rb'))
    sample_zlow = pickle.load(open('../data/LAEfrac_saves/sample_zlow_msaexp_pypeit.sav', 'rb'))
    sample_zhi = pickle.load(open('../data/LAEfrac_saves/sample_zhi_msaexp_pypeit.sav', 'rb'))
    sample_lit = pickle.load(open('../data/LAEfrac_saves/sample_literature.sav', 'rb'))

    flatsample_J1148_zQ = sample_J1148_zQ.get_chain(discard=1000, thin=20,flat=True)
    EWfrac_J1148_zQ = EWfrac_from_MCMC(EWlim, flatsample_J1148_zQ)

    flatsample_J0100_zQ = sample_J0100_zQ.get_chain(discard=1000, thin=20,flat=True)
    EWfrac_J0100_zQ = EWfrac_from_MCMC(EWlim, flatsample_J0100_zQ)
    
    flatsample_zlo = sample_zlow.get_chain(discard=1000, thin=20,flat=True)
    EWfrac_zlo = EWfrac_from_MCMC(EWlim, flatsample_zlo)
    
    flatsample_zhi = sample_zhi.get_chain(discard=1000, thin=20,flat=True)
    EWfrac_zhi = EWfrac_from_MCMC(EWlim, flatsample_zhi)
    
    flatsample_lit = sample_lit.get_chain(discard=1000, thin=20,flat=True)
    EWfrac_lit = EWfrac_from_MCMC(EWlim, flatsample_lit)
    
    fig, ax = plt.subplots()

    percentiles_J1148_Q = np.percentile(EWfrac_J1148_zQ, [16,50,84])
    ax.errorbar([6.42], [percentiles_J1148_Q[1]], \
                yerr=[[percentiles_J1148_Q[1]-percentiles_J1148_Q[0]], [percentiles_J1148_Q[2]-percentiles_J1148_Q[1]]],\
                fmt='o', color='darkorange', label=r'J1148, $|z_Q-z|<0.05$')
    
    percentiles_J0100_Q = np.percentile(EWfrac_J0100_zQ, [16,50,84])
    ax.errorbar([6.327], [percentiles_J0100_Q[1]], \
                yerr=[[percentiles_J0100_Q[1]-percentiles_J0100_Q[0]], [percentiles_J0100_Q[2]-percentiles_J0100_Q[1]]],\
                fmt='s', color='darkred', label='J0100, $|z_Q-z|<0.05$')

    zlow = np.mean(allinfo_zlow['redshift'])
    percentiles_zlo = np.percentile(EWfrac_zlo, [16,50,84])
    ax.errorbar([zlow], [percentiles_zlo[1]], \
                yerr=[[percentiles_zlo[1]-percentiles_zlo[0]], [percentiles_zlo[2]-percentiles_zlo[1]]],\
                fmt='o', color='k', mfc='none', label='Foreground/Background')
    
    zhi = np.mean(allinfo_zhi['redshift'])
    percentiles_zhi = np.percentile(EWfrac_zhi, [16,50,84])
    ax.errorbar([zhi], [percentiles_zhi[1]], \
                yerr=[[percentiles_zhi[1]-percentiles_zhi[0]], [percentiles_zhi[2]-percentiles_zhi[1]]],\
                fmt='o', color='k', mfc='none')

    # literature
    ax.plot([6,7], [np.mean(EWfrac_lit)]*2, 'k--', label=r'JWST 6<z<7 (Kageura+25)')
    print(np.percentile(EWfrac_lit,84))
    ax.fill_between([6,7], [np.percentile(EWfrac_lit,84)]*2, [np.percentile(EWfrac_lit,16)]*2, color='k', alpha=0.2)
    ax.set_xlim([6,7])

    ax.legend()
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$f (EW>10\AA)$', fontsize=14)

    ax.tick_params(axis='both', direction='in')
    plt.savefig('./Efrac_10AA.pdf')

    plt.show()


def save_table_J1148():
    # RA Dec
    tbl_J1148 = Table.read('../../ID3117/pypeit-reduc/data-release-2/masterinfo_J1148_msaexp_pypeit.fits')
    tbl_J1148_lya = Table.read('./EW_fesc_J1148_msaexp_pypeit.fits')

#    tbl_J1148 = tbl_J1148[tbl_J1148['redshift']>5.5]
    tbl_J1148_good = select_good_sources_to_plot(tbl_J1148_lya,\
                                                 'J1148', Muvlim=[-24,-19.])

    IDlist = []
    RAlist = []
    Declist = []
    zsyslist = []
    zlyalist = []
    zlya_ue = []
    zlya_le = []
    Flyalist = []
    Flya_ue = []
    Flya_le = []
    EWlyalist = []
    EWlya_ue = []
    EWlya_le = []
    Muvlist = []
    fesclist = []
    fesc_err = []

    for index in range(len(tbl_J1148_good)):
        number = tbl_J1148_good['NUMBER'][index]
        print(number)
        subtbl1 = tbl_J1148[tbl_J1148['isource']=='3117_%d'%number]
        print(subtbl1)
        IDlist.append('J1148_%d'%number)
        RAlist.append(subtbl1['RA'][0])
        Declist.append(subtbl1['Dec'][0])
        zsyslist.append(tbl_J1148_good['z_O3doublet_combined_n'][index])
        Muvlist.append(tbl_J1148_good['Muv'][index])

        zlyalist.append(tbl_J1148_good['zlyaquantile'][index][3])
        zlya_ue.append(tbl_J1148_good['zlyaquantile'][index][4]-\
                       tbl_J1148_good['zlyaquantile'][index][3])
        zlya_le.append(tbl_J1148_good['zlyaquantile'][index][3]-\
                       tbl_J1148_good['zlyaquantile'][index][2])

        # tricky ones
        if tbl_J1148_good['EWsigma'][index]>0:
            Flyalist.append(tbl_J1148_good['fluxquantile'][index][3])
            Flya_ue.append(tbl_J1148_good['fluxquantile'][index][4]-\
                            tbl_J1148_good['fluxquantile'][index][3])
            Flya_le.append(tbl_J1148_good['fluxquantile'][index][3]-\
                            tbl_J1148_good['fluxquantile'][index][2])

            EWlyalist.append(tbl_J1148_good['EWquantile'][index][3])
            EWlya_ue.append(tbl_J1148_good['EWquantile'][index][4]-\
                            tbl_J1148_good['EWquantile'][index][3])
            EWlya_le.append(tbl_J1148_good['EWquantile'][index][3]-\
                            tbl_J1148_good['EWquantile'][index][2])

            fesclist.append(tbl_J1148_good['fescmean'][index])
            fesc_err.append(tbl_J1148_good['fescsigma'][index])

        else:
            if tbl_J1148_good['fluxquantile'][index][-1]<0:
                Flyalist.append((tbl_J1148_good['fluxquantile'][index][4]-\
                                tbl_J1148_good['fluxquantile'][index][2])/2)
            else:
                Flyalist.append(tbl_J1148_good['fluxquantile'][index][-1])
            Flya_ue.append(-1)
            Flya_le.append(-1)

            EWlyalist.append(tbl_J1148_good['EWmean'][index])
            EWlya_ue.append(-1)
            EWlya_le.append(-1)

            fesclist.append(tbl_J1148_good['fescmean'][index])
            fesc_err.append(-1)


    tbl_save = Table(dict(ID = IDlist,
    RA = RAlist,
    Dec = Declist,
    zsys = zsyslist,
    zlya = zlyalist,
    zlya_ue = zlya_ue,
    zlya_le = zlya_le,
    Flya = Flyalist,
    Flya_ue = Flya_ue,
    Flya_le = Flya_le,
    EWlya = EWlyalist,
    EWlya_ue = EWlya_ue,
    EWlya_le = EWlya_le,
    Muv = Muvlist,
    fesc = fesclist,
    fesc_err = fesc_err
))

    print(tbl_J1148_good)
    tbl_save = tbl_save[tbl_save['zsys']>6]
    print(len(tbl_save))
    tbl_save.write('finaltbl_J1148.fits', overwrite=True)

def save_table_J0100():

    tbl_J0100 = Table.read('../../ID4713/pypeit-reduc/data-release-2/masterinfo_J0100_pypeit_msaexp_update.fits')
    tbl_J0100_lya = Table.read('./EW_fesc_J0100_msaexp_pypeit.fits')

    tbl_J0100_good = select_good_sources_to_plot(tbl_J0100_lya,\
                                                 'J0100', Muvlim=[-24,-19.])

    IDlist = []
    RAlist = []
    Declist = []
    zsyslist = []
    zlyalist = []
    zlya_ue = []
    zlya_le = []
    Flyalist = []
    Flya_ue = []
    Flya_le = []
    EWlyalist = []
    EWlya_ue = []
    EWlya_le = []
    Muvlist = []
    fesclist = []
    fesc_err = []

    for index in range(len(tbl_J0100_good)):
        number = tbl_J0100_good['NUMBER'][index]
        print(number)
        subtbl1 = tbl_J0100[tbl_J0100['isource']=='4713_%d'%number]
        print(subtbl1)
        IDlist.append('J0100_%d'%number)
        RAlist.append(subtbl1['RA'][0])
        Declist.append(subtbl1['Dec'][0])
        zsyslist.append(subtbl1['redshift'][0])
        Muvlist.append(tbl_J0100_good['Muv'][index])

        zlyalist.append(tbl_J0100_good['zlyaquantile'][index][3])
        zlya_ue.append(tbl_J0100_good['zlyaquantile'][index][4]-\
                       tbl_J0100_good['zlyaquantile'][index][3])
        zlya_le.append(tbl_J0100_good['zlyaquantile'][index][3]-\
                       tbl_J0100_good['zlyaquantile'][index][2])

        # tricky ones
        if tbl_J0100_good['EWsigma'][index]>0:
            Flyalist.append(tbl_J0100_good['fluxquantile'][index][3])
            Flya_ue.append(tbl_J0100_good['fluxquantile'][index][4]-\
                            tbl_J0100_good['fluxquantile'][index][3])
            Flya_le.append(tbl_J0100_good['fluxquantile'][index][3]-\
                            tbl_J0100_good['fluxquantile'][index][2])

            EWlyalist.append(tbl_J0100_good['EWquantile'][index][3])
            EWlya_ue.append(tbl_J0100_good['EWquantile'][index][4]-\
                            tbl_J0100_good['EWquantile'][index][3])
            EWlya_le.append(tbl_J0100_good['EWquantile'][index][3]-\
                            tbl_J0100_good['EWquantile'][index][2])

            fesclist.append(tbl_J0100_good['fescmean'][index])
            fesc_err.append(tbl_J0100_good['fescsigma'][index])

        else:
            if tbl_J0100_good['fluxquantile'][index][-1]<0:
                Flyalist.append((tbl_J0100_good['fluxquantile'][index][4]-\
                                tbl_J0100_good['fluxquantile'][index][2])/2)
            else:
                Flyalist.append(tbl_J0100_good['fluxquantile'][index][-1])

            Flya_ue.append(-1)
            Flya_le.append(-1)

            EWlyalist.append(tbl_J0100_good['EWmean'][index])
            EWlya_ue.append(-1)
            EWlya_le.append(-1)

            fesclist.append(tbl_J0100_good['fescmean'][index])
            fesc_err.append(-1)


    tbl_save = Table(dict(ID = IDlist,
    RA = RAlist,
    Dec = Declist,
    zsys = zsyslist,
    zlya = zlyalist,
    zlya_ue = zlya_ue,
    zlya_le = zlya_le,
    Flya = Flyalist,
    Flya_ue = Flya_ue,
    Flya_le = Flya_le,
    EWlya = EWlyalist,
    EWlya_ue = EWlya_ue,
    EWlya_le = EWlya_le,
    Muv = Muvlist,
    fesc = fesclist,
    fesc_err = fesc_err
))

#    plt.hist(tbl_save['zsys'])
#    plt.show()
    tbl_save = tbl_save[tbl_save['zsys']>6]
    tbl_save.write('finaltbl_J0100.fits', overwrite=True)


def print_table(tblfile, output=None):

    tbl = Table.read(tblfile)
    for index in range(len(tbl)):
        string = ''
        string += f'{tbl['ID'][index].replace('_', '\_')} & '
        string += '%.6f & '%tbl['RA'][index]
        string += '%.6f & '%tbl['Dec'][index]
        string += '%.2f & '%tbl['zsys'][index]
        string += '%.2f & '%tbl['Muv'][index]

        if tbl['EWlya_ue'][index]>0:
            string += '$%.3f^{+%.3f}_{-%.3f}$ & '%(tbl['zlya'][index]-1,\
                                tbl['zlya_ue'][index], tbl['zlya_le'][index])
            string += '$%.2f^{+%.2f}_{-%.2f}$ & '%(tbl['Flya'][index],\
                            tbl['Flya_ue'][index], tbl['Flya_le'][index])
            string += '$%.2f^{+%.2f}_{-%.2f}$ & '%(tbl['EWlya'][index],\
                            tbl['EWlya_ue'][index], tbl['EWlya_le'][index])
            string += r'$%.2f\pm{%.2f}$ \\'%(tbl['fesc'][index], tbl['fesc_err'][index])
        else:
            string += ' & '
            string += '$<%.2f$ & '%(tbl['Flya'][index])
            string += '$<%.2f$ & '%(tbl['EWlya'][index])
            string += r'$<%.2f$ \\'%(tbl['fesc'][index])


        print(string)


def save_table_z67(field):
    tbl = Table.read(f'./finaltbl_{field}.fits')
    tbl_z67 = tbl[(tbl['zsys']>6)&(tbl['zsys']<7)]
    tbl_z67.write(f'./finaltbl_{field}_z67.fits', overwrite=True)
    print(tbl_z67)
    print(len(tbl_z67))

if __name__=='__main__':
#    save_table_z67('J0100')
#    save_table_z67('J1148')
#    save_table_J1148()
    plot_redshift_EW(Muv_min=-24, Muv_max=-19.)
#    plot_redshift_fesc(Muv_min=-24, Muv_max=-19.)

#    plot_field_J0100()
#    plot_field_J1148_talk()
#    plot_spec_examples_talk()
#    plot_LAE_frac(10)
#    print_table('./finaltbl_J1148.fits')
