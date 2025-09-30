'''
Colleting information for all the targets from various sources, and compile
them into a master table that I will put online.
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
from scipy.stats import lognorm, norm

import LAE_EW_utils as LAEew
import LAE_spec_utils as LAEspec

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import bilby
from dynesty import utils as dyfunc

import extinction

import warnings
warnings.filterwarnings("ignore")

def add_EW_and_fesc(linefittbl, outdir, tbl_prosp):
    '''
    Compute the Lya EW and Lya fesc values and errors, and append it to the 
    input table.

    Parameters
    ----------
    linefittbl : astropy.table.Table
        The table with Lya line fitting information

    outdir : str
        The directory where bilby fits are stored

    tbl_prosp : astropy.table.Table
        The table with prospector fitting results

    Returns
    -------
    tbl: astropy.table.Table
        The table with EW and fesc values and errors added.
    '''
    tbl = linefittbl.copy()

    # the four columns that need to be added
    EWmean = []
    EWsigma = []
    fescmean = []
    fescsigma = []

    avlist = []

    # determine which field name and program ID we will work on
    if 'J0100' in outdir:
        field = 'J0100'
        prog_id = '4713'
    elif 'J1148' in outdir:
        field = 'J1148'
        prog_id = '3117'

    # loop all the table rows to compute the quantities to be added

    for index in range(len(tbl)):
        isource = prog_id+'_%d'%(tbl['NUMBER'][index])
        label = isource

        # systemic (O3) redshift
        z = tbl['z_O3doublet_combined_n'][index]

        # read the fitting result for this target
        result = bilby.result.read_in_result(outdir=outdir, label=label)
        samples = result.nested_samples
        weights = np.array(samples['weights'])
        flux_sample = dyfunc.resample_equal(np.array(samples['flux']), weights)
        A_sample = dyfunc.resample_equal(np.array(samples['A']), weights)
        EWsample = flux_sample / A_sample / z

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
            # this means a 3sigma detection
            EWmean.append(np.median(EWsample))
            EWsigma.append(np.std(EWsample))
        elif tbl['fluxquantile'][index][-1]<0:
            # this means a significant negative Lya flux
            EWmean.append(np.std(EWsample))
            # a sigma value of -1 means an upper limit
            EWsigma.append(-1)
        else:
            # this means the flux is consistent with 0 within 3sigma
            EWmean.append(tbl['EWquantile'][index][-1])
            EWsigma.append(-1)

        # compute the expected lya EW (when fesc=1) using the Hbeta line
        EW_from_Hb, eEW_from_Hb = \
            LAEspec.EW_Lya_from_NIRCam(fHb, fHb_err, mags, magerrs, z)

        # correct for extinction
        subtbl = tbl_prosp[(tbl_prosp['FIELD']==field)&\
                           (tbl_prosp['NUMBER']==tbl['NUMBER'][index])]
        if len(subtbl)>0:
            # the algorithm from Jorryt's table
            # assumes Av/E(B-V)=3.438
            av = float(subtbl['EBV'][0]) * 3.43816749
            ahb = extinction.calzetti00(np.array([4863.0]), av, 3.43)
            hbcorr = 10**(0.4*ahb[0])
            avlist.append(av)
            #print(hbcorr)
        else:
            hbcorr = 1
            avlist.append(1)

        EW_from_Hb_corr = EW_from_Hb * hbcorr# non-extincted hb
        eEW_from_Hb_corr = eEW_from_Hb * hbcorr
        if isource == '3117_2637':
            print(isource, EW_from_Hb_corr, av)

        fescmean.append(EWmean[-1] / EW_from_Hb_corr)

        if EWsigma[-1]<0:
            # this means a non-detection
            fescsigma.append(-1)
        else:
            # this means a detection
            relative_err = np.sqrt(np.std(EWsample)**2/np.median(EWsample)**2\
                                     + (eEW_from_Hb/EW_from_Hb)**2)
            fescsigma.append(fescmean[-1] * relative_err)

    tbl['EWmean'] = np.array(EWmean)
    tbl['EWsigma'] = np.array(EWsigma)
    tbl['fescmean'] = np.array(fescmean)
    tbl['fescsigma'] = np.array(fescsigma)
    tbl['av'] = np.array(avlist)

    return tbl


def save_table_one_field(field='J1148'):
    # We need to get RA and Dec from this Table
#    '''
    if field=='J1148':
        Prog_ID = 3117
        bad_ids = [6406, 9192, 16089, 15411]
#        AGN_ids = [3607, 9314, 11709]
    elif field=='J0100':
        Prog_ID = 4713
        bad_ids = [812, 452, 1228, 2159]
#        AGN_ids = [10287, 9970, 7230]
#    '''
    tbl_pypeit = Table.read(f'../../ID{Prog_ID}/pypeit-reduc/data-release-2/masterinfo_{field}_msaexp_pypeit.fits')

    # this table is generated using the add_EW_and_fesc() function above
    tbl_lya = Table.read(f'./EW_fesc_{field}_msaexp_pypeit_beta_civ_av.fits')

    # exclude targets with obviously bad properties

    # these are either with a bright source nearby or with uncovered lya
    # so we exclude them
    # we also want a specific Muv range and redshift range
    tbl_good = tbl_lya[\
            (~np.isin(tbl_lya['NUMBER'], bad_ids))&\
            (tbl_lya['Muv']<-19)&\
            (tbl_lya['z_O3doublet_combined_n']<7)&\
            (tbl_lya['z_O3doublet_combined_n']>6)]

    # These are the columns we want to compile

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
    AGN = []

    EWmean = []
    EWsigma = []
    fescmean = []
    fescsigma = []

    zcivlist = []
    zciv_ue = []
    zciv_le = []
    Fcivlist = []
    Fciv_ue = []
    Fciv_le = []
    EWcivlist = []
    EWciv_ue = []
    EWciv_le = []

    avlist = []

    for index in range(len(tbl_good)):
        number = tbl_good['NUMBER'][index]

        # to find the RA and Dec
        subtbl1 = tbl_pypeit[tbl_pypeit['isource']=='%d_%d'%(Prog_ID, number)]
        IDlist.append('%s_%d'%(field, number))
        RAlist.append(subtbl1['RA'][0])
        Declist.append(subtbl1['Dec'][0])

        # we can then find the other properties
        zsyslist.append(tbl_good['z_O3doublet_combined_n'][index])
        Muvlist.append(tbl_good['Muv'][index])

        zlyalist.append(tbl_good['zlyaquantile'][index][3])
        zlya_ue.append(tbl_good['zlyaquantile'][index][4]-\
                       tbl_good['zlyaquantile'][index][3])
        zlya_le.append(tbl_good['zlyaquantile'][index][3]-\
                       tbl_good['zlyaquantile'][index][2])

        # tricky ones
        if tbl_good['EWsigma'][index]>0:
            Flyalist.append(tbl_good['fluxquantile'][index][3])
            Flya_ue.append(tbl_good['fluxquantile'][index][4]-\
                            tbl_good['fluxquantile'][index][3])
            Flya_le.append(tbl_good['fluxquantile'][index][3]-\
                            tbl_good['fluxquantile'][index][2])

            EWlyalist.append(tbl_good['EWquantile'][index][3])
            EWlya_ue.append(tbl_good['EWquantile'][index][4]-\
                            tbl_good['EWquantile'][index][3])
            EWlya_le.append(tbl_good['EWquantile'][index][3]-\
                            tbl_good['EWquantile'][index][2])

            fesclist.append(tbl_good['fescmean'][index])
            fesc_err.append(tbl_good['fescsigma'][index])

        else:
            if tbl_good['fluxquantile'][index][-1]<0:
                Flyalist.append((tbl_good['fluxquantile'][index][4]-\
                                tbl_good['fluxquantile'][index][2])/2)
            else:
                Flyalist.append(tbl_good['fluxquantile'][index][-1])
            Flya_ue.append(-1)
            Flya_le.append(-1)

            EWlyalist.append(tbl_good['EWmean'][index])
            EWlya_ue.append(-1)
            EWlya_le.append(-1)

            fesclist.append(tbl_good['fescmean'][index])
            fesc_err.append(-1)

        # deal with CIV

        if tbl_good['fcivquantile'][index][0]>0:
            Fcivlist.append(tbl_good['fcivquantile'][index][3])
            Fciv_ue.append(tbl_good['fcivquantile'][index][4]-\
                            tbl_good['fcivquantile'][index][3])
            Fciv_le.append(tbl_good['fcivquantile'][index][3]-\
                            tbl_good['fcivquantile'][index][2])

            EWcivlist.append(tbl_good['ewcivquantile'][index][3])
            EWciv_ue.append(tbl_good['ewcivquantile'][index][4]-\
                            tbl_good['ewcivquantile'][index][3])
            EWciv_le.append(tbl_good['ewcivquantile'][index][3]-\
                            tbl_good['ewcivquantile'][index][2])

        else:
            Fcivlist.append(tbl_good['fcivquantile'][index][-1])
            Fciv_ue.append(-1)
            Fciv_le.append(-1)

            EWcivlist.append(tbl_good['ewcivquantile'][index][-1])
            EWciv_ue.append(-1)
            EWciv_le.append(-1)

        EWmean.append(tbl_good['EWmean'][index])
        EWsigma.append(tbl_good['EWsigma'][index])
        fescmean.append(tbl_good['fescmean'][index])
        fescsigma.append(tbl_good['fescsigma'][index])

        avlist.append(tbl_good['av'][index])

        print(number, Fcivlist[-1], EWcivlist[-1], EWciv_ue[-1], EWmean[-1], EWsigma[-1])
        if tbl_good['fcivquantile'][index][0]>0 and\
                tbl_good['ewcivquantile'][index][3]>12:
            AGN.append(True)

        else:
            AGN.append(False)

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
    fesc_err = fesc_err,
    AGN = AGN,
    EWmean = EWmean,
    EWsigma = EWsigma,
    fescmean = fescmean,
    fescsigma = fescsigma,
    av = avlist
))

    print(len(tbl_save))
    tbl_save.write('final_sample_%s.fits'%field, overwrite=True)


def print_table(tblfile, output=None):

    tbl = Table.read(tblfile)
    for index in range(len(tbl)):
        agn = tbl['AGN'][index]
        if not agn:
            continue

        string = ''
        string += f'{tbl['ID'][index].replace('_', '\_')} & '
        string += '%.6f & '%tbl['RA'][index]
        string += '%.6f & '%tbl['Dec'][index]
        string += '%.3f & '%tbl['zsys'][index]
        string += '-%.2f & '%tbl['Muv'][index]

        if tbl['EWlya_ue'][index]>0:
            string += '$%.4f^{+%.4f}_{-%.4f}$ & '%(tbl['zlya'][index]-1,\
                                tbl['zlya_ue'][index], tbl['zlya_le'][index])
            string += '$%.2f^{+%.2f}_{-%.2f}$ & '%(tbl['Flya'][index],\
                            tbl['Flya_ue'][index], tbl['Flya_le'][index])
            string += '$%.2f^{+%.2f}_{-%.2f}$ & '%(tbl['EWlya'][index],\
                            tbl['EWlya_ue'][index], tbl['EWlya_le'][index])
            string += r'$%.3f\pm{%.3f}$ \\'%(tbl['fesc'][index], tbl['fesc_err'][index])
        else:
            string += ' & '
            string += '$<%.2f$ & '%(tbl['Flya'][index])
            string += '$<%.2f$ & '%(tbl['EWlya'][index])
            string += r'$<%.3f$ \\'%(tbl['fesc'][index])


        print(string)


def save_table_z67(field):
    tbl = Table.read(f'./finaltbl_{field}.fits')
    tbl_z67 = tbl[(tbl['zsys']>6)&(tbl['zsys']<7)]
    tbl_z67.write(f'./finaltbl_{field}_z67.fits', overwrite=True)
    print(tbl_z67)
    print(len(tbl_z67))


if __name__=='__main__':
#    save_table_one_field('J0100')
    # test the add_EW_and_fesc() function
    '''
    allinfo_J0100 = Table.read('../data/lineinfo_J0100_msaexp_pypeit_v2.fits')
    allinfo_J1148 = Table.read('../data/lineinfo_J1148_msaexp_pypeit_v2.fits')

    tbl_prosp = Table.read('../data/EIGER_5fields_O3doublets_SYSTEMS_28022024_withProsp.fits')

    outdir_J0100 = '../data/lya_saves/J0100/bilby_fit_lya_msaexp_pypeit_v2/'
    outdir_J1148 = '../data/lya_saves/J1148/bilby_fit_lya_msaexp_pypeit_v2/'

    tbl_ewfesc_J0100 = add_EW_and_fesc(allinfo_J0100, outdir_J0100, tbl_prosp)
    tbl_ewfesc_J0100.write('./EW_fesc_J0100_msaexp_pypeit_beta_civ_av.fits',\
                           overwrite=True)
    tbl_ewfesc_J1148 = add_EW_and_fesc(allinfo_J1148, outdir_J1148, tbl_prosp)
    tbl_ewfesc_J1148.write('./EW_fesc_J1148_msaexp_pypeit_beta_civ_av.fits',\
                           overwrite=True)

    save_table_one_field('J0100')
    save_table_one_field('J1148')

    '''
    print_table('./final_sample_J1148.fits')
    print_table('./final_sample_J0100.fits')
