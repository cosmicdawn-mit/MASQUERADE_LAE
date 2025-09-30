import os, sys
import numpy as np
import glob
import matplotlib.pyplot as plt

from astropy.table import Table, vstack, hstack
from astropy import units
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


from scipy.optimize import minimize, leastsq
import bilby
from dynesty import utils as dyfunc

from mylib.spectrum.spec_measurement import Spectrum, read_filter, instru_filter_dict

# this exception means that the emission line to be fitted is not covered
class LineNotCoveredError(Exception):
    pass

# change it to where this file locates
datadir = os.getcwd()


F115Wdata = np.loadtxt(datadir+'F115W_mean_system_throughput.txt')
F200Wdata = np.loadtxt(datadir+'F200W_mean_system_throughput.txt')

np.int = int

# functions for Muv
def integrated_flux(spec, filtercurve):
    # computing the integrated transmission flux given the flter curve and the spectrum
    wave = spec[:,0]
    flux = spec[:,1]
    fwave = filtercurve[:,0] * 1e4
    fthru = filtercurve[:,1]

    interpthru = np.interp(wave, fwave, fthru)
    intflux = np.trapz(interpthru*flux, wave)

    return intflux

def mags_from_UV_param_simple_for_Lya(params, redshift, filtercurve):
    # given the power-law parameters and the redshift, compute the magnitudes
    # A: the intensity at restframe 1500 AA
    # beta: the UV slope
    A, beta = params
    wave = filtercurve[:,0]*1e4

    # power law spec
    mdlflux = A * (wave/1215.67/(1+redshift))**beta * 1e-18
    mdlspec = np.transpose([wave, mdlflux])

    # standard spec is 3631 Jy
    speed_of_light = 299792458 * 1e10#angstrom/s
    stdflux = 3631 * 1e-23 * speed_of_light /wave**2
    stdspec = np.transpose([wave, stdflux])

    intflux = integrated_flux(mdlspec, filtercurve)
    zpflux = integrated_flux(stdspec, filtercurve)

    return -2.5 * np.log10(intflux/zpflux)


def mags_from_UV_param_simple(params, redshift, filtercurve):
    # given the power-law parameters and the redshift, compute the magnitudes
    # A: the intensity at restframe 1500 AA
    # beta: the UV slope
    A, beta = params
    wave = filtercurve[:,0]*1e4

    # power law spec
    mdlflux = A * (wave/1500/(1+redshift))**beta * 1e-18
    mdlspec = np.transpose([wave, mdlflux])

    # standard spec is 3631 Jy
    speed_of_light = 299792458 * 1e10#angstrom/s
    stdflux = 3631 * 1e-23 * speed_of_light /wave**2
    stdspec = np.transpose([wave, stdflux])

    intflux = integrated_flux(mdlspec, filtercurve)
    zpflux = integrated_flux(stdspec, filtercurve)

    return -2.5 * np.log10(intflux/zpflux)

def residual_mags(params, mags, magerrs, redshift):
    # the residual used when fitting the UV parameters using the mags
    mdlF115W = mags_from_UV_param_simple(params, redshift, F115Wdata)
    mdlF200W = mags_from_UV_param_simple(params, redshift, F200Wdata)
    mdlmags = np.array([mdlF115W, mdlF200W])

#    return np.sum((mdlmags-mags)**2/magerrs**2)
    return (mdlmags-mags)/magerrs

def residual_mags_for_Lya(params, mags, magerrs, redshift):
    # the residual used when fitting the UV parameters using the mags
    mdlF115W = mags_from_UV_param_simple_for_Lya(params, redshift, F115Wdata)
    mdlF200W = mags_from_UV_param_simple_for_Lya(params, redshift, F200Wdata)
    mdlmags = np.array([mdlF115W, mdlF200W])

    return (mdlmags-mags)/magerrs


def fit_UV_params(mags, magerrs, redshift):
    # fit the UV flux and slope. May want to change to MCMC.
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    x0 = [0.2,-2]
#    res = minimize(residual_mags, x0, args=(mags, magerrs, redshift),\
#                   method='TNC', bounds=((1e-3,None),(-4,1)))

    res = leastsq(residual_mags, x0, args=(mags, magerrs, redshift),\
                   full_output=True)

    return res


def fit_UV_params_for_Lya(mags, magerrs, redshift):
    # fit the UV flux and slope. May want to change to MCMC.
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    x0 = [0.2,-2]

    res = leastsq(residual_mags_for_Lya, x0, args=(mags, magerrs, redshift),\
                   full_output=True)

    return res

def get_Muv_from_mag(mags, magerrs, redshift, cosmo):
    # estimate Muv at 1500AA given the F115W and F200W magnitudes
    # assuming a power law

    result = fit_UV_params(mags, magerrs, redshift)
    A, beta = result[0]
    Aerr, betaerr = np.sqrt(np.diag(result[1]))

    wavegrid = np.arange(1200,1600)*(1+redshift)
    flam = A * (wavegrid/1500/(1+redshift))**beta

    spec = Spectrum(wavelength=wavegrid, value=flam*1e-18)
    spec.to_abs(redshift=redshift)
    Flam = spec.getvalue(1500)

    speed_of_light = 299792458 * 1e10#angstrom/s
    Fnu = Flam * 1500 * 1500 / speed_of_light# erg/s/cm^2/Hz
    return -2.5*np.log10(Fnu/(1e-23*3631))

def EW_Lya_from_NIRCam(f_Hb, e_f_Hb, mags, magerrs, redshift):
    result = fit_UV_params_for_Lya(mags, magerrs, redshift)
    A, beta = result[0]
    Aerr, betaerr = np.sqrt(np.diag(result[1]))
    #wave_lya = 1215.67 * (1+redshift)
    cont_lya = A
    # * (wave_lya/1500/(1+redshift)) ** beta#in 1e-18

    # a correction factor of 25, following Tang+24
    # https://iopscience.iop.org/article/10.3847/1538-4357/ad7eb7/pdf
    EW_from_Hb = f_Hb/cont_lya/(1+redshift)* 25
    eEW_from_Hb_1 = e_f_Hb/cont_lya/(1+redshift)* 25
    eEW_from_Hb_2 = EW_from_Hb * Aerr/A

    eEW_from_Hb = np.sqrt(eEW_from_Hb_1**2+eEW_from_Hb_2**2)
    
    return EW_from_Hb, eEW_from_Hb

def lya_profile_simple(wave, center, fwhm, flux, A, beta, wave0):
    #wave0 = other_param
    sigma = fwhm/3e5 * center / 2 / np.sqrt(2*np.log(2))
    line = 1/sigma/np.sqrt(2*np.pi) * np.exp(-(wave-center)**2/2/sigma**2)

    # wave0 is the systematic Lya wavelength. beta the UV slope.\
    # A the cont normalization.
    cont = A * (wave/wave0)**beta
    cont[wave<wave0]=0

    totalflux = line * flux + cont

    return totalflux

def emline_profile_simple(wave, center, fwhm, flux, A, beta, wave0):
    sigma = fwhm/3e5 * center / 2 / np.sqrt(2*np.log(2))
    line = 1/sigma/np.sqrt(2*np.pi) * np.exp(-(wave-center)**2/2/sigma**2)

    # wave0 is the systematic Lya wavelength. beta the UV slope.\
    # A the cont normalization.
    cont = A * (wave/wave0)**beta

    totalflux = line * flux + cont

    return totalflux

class LyaFit(object):
    def __init__(self, wave, flux, error, redshift, outdir, label):
        self.wave = wave
        self.flux = flux
        self.error = error
        self.zO3 = redshift

        self.outdir = outdir
        self.label = label

    def set_likelihood(self, func):
        self.likelihood = bilby.likelihood.GaussianLikelihood(\
                self.wave, self.flux, func, self.error)
'''
class LyaFitSimple(LyaFit):
    def __init__(self, wave, flux, error, redshift, outdir, label):
        super.__init__(wave, flux, error, redshift, outdir, label)
        self.set_likelihood(lya_profile_simple)

    def set_priors(self):
        restwave = 1215.67
        priors = dict()

        wave0 = (1+self.zO3)*restwave
        fwhm0 = 500
        amp0 = np.max(self.flux) * fwhm0 * np.sqrt(2*np.pi) / 2.355 / 3e5 * wave0
        x0 = [wave0, fwhm0, amp0, 0]
        value_extreme = np.max(np.abs(self.flux))

        priors["center"] = bilby.core.prior.Uniform(restwave*(1+self.zO3)*(1-1/300), \
                                                    restwave*(1+self.zO3)*(1+1/300), "center")
        priors["fwhm"] = bilby.core.prior.Uniform(300, 1000, "fwhm")
        priors["flux"] = bilby.core.prior.Uniform(-amp0*10, amp0*10, "flux")
        priors["A"] = bilby.core.prior.Uniform(-value_extreme*2, value_extreme*2, "A")
        priors["beta"] = bilby.core.prior.Uniform(-4, 0, "beta")
        priors["wave0"] = bilby.core.prior.DeltaFunction(wave0, "wave0")

    def draw_uniform_sample(self):
        samples = self.lya_result.nested_samples
        weights = np.array(samples['weights'])

        # get the following properties:
        self.center_sample = dyfunc.resample_equal(np.array(samples['center']), weights)
        self.fwhm_sample = dyfunc.resample_equal(np.array(samples['fwhm']), weights)
        self.flux_sample = dyfunc.resample_equal(np.array(samples['flux']), weights)
        self.A_sample = dyfunc.resample_equal(np.array(samples['A']), weights)
        self.beta_sample = dyfunc.resample_equal(np.array(samples['beta']), weights)

        self.EW_sample = self.flux_sample / self.A_sample


class LyaFitSkewed(LyaFit):
    def __init__(self, wave, flux, error, redshift, outdir, label):
        super.__init__(wave, flux, error, redshift, outdir, label)
        self.set_likelihood(lya_profile_simple)

    def set_priors(self):
        restwave = 1215.67
        priors = dict()

        wave0 = (1+self.zO3)*restwave
        fwhm0 = 500
        amp0 = np.max(self.flux) * fwhm0 * np.sqrt(2*np.pi) / 2.355 / 3e5 * wave0
        x0 = [wave0, fwhm0, amp0, 0]
        value_extreme = np.max(np.abs(self.flux))

        priors["center"] = bilby.core.prior.Uniform(restwave*(1+self.zO3)*(1-1/300), \
                                                    restwave*(1+self.zO3)*(1+1/300), "center")
        priors["fwhm"] = bilby.core.prior.Uniform(300, 1000, "fwhm")
        priors["flux"] = bilby.core.prior.Uniform(-amp0*10, amp0*10, "flux")
        priors["A"] = bilby.core.prior.Uniform(-value_extreme*2, value_extreme*2, "A")
        priors["beta"] = bilby.core.prior.Uniform(-4, 0, "beta")
        priors["wave0"] = bilby.core.prior.Uniform(restwave*(1+self.zO3)*(1-1/300), \
                                                    restwave*(1+self.zO3)*(1+1/300), "wave0")

    def draw_uniform_sample(self):
        samples = self.lya_result.nested_samples
        weights = np.array(samples['weights'])

        # get the following properties:
        self.center_sample = dyfunc.resample_equal(np.array(samples['center']), weights)
        self.fwhm_sample = dyfunc.resample_equal(np.array(samples['fwhm']), weights)
        self.flux_sample = dyfunc.resample_equal(np.array(samples['flux']), weights)
        self.A_sample = dyfunc.resample_equal(np.array(samples['A']), weights)
        self.beta_sample = dyfunc.resample_equal(np.array(samples['beta']), weights)

        self.EW_sample = self.flux_sample / self.A_sample
'''

class Spectrum1D(object):

    def prepare_data_for_fit(self, restwave, **kwargs):
        minwave = kwargs.pop('minwave', (1+self.zO3)*0.9*restwave)
        maxwave = kwargs.pop('maxwave', (1+self.zO3)*1.1*restwave)
        mask_wavelength = (self.wave>minwave)&(self.wave<maxwave)&self.mask

        wave_masked = self.wave[mask_wavelength]
        flux_masked = self.flam[mask_wavelength]
        err_masked = self.err_flam[mask_wavelength]

        narrow_lya_range_flux = flux_masked[(wave_masked>(1+self.zO3)*0.99*restwave)&\
                                            (wave_masked<(1+self.zO3)*1.01*restwave)]

        if len(wave_masked)==0 or np.all(narrow_lya_range_flux==0):
            raise LineNotCoveredError(f'Line not covered, len(wave_masked)={len(wave_masked)}, zero flux = {np.all(narrow_lya_range_flux==0)}')

        return wave_masked, flux_masked, err_masked

    def fit_lya_profile_skewed_bilby(self):
        restwave = 1215.67
        wave_masked, flux_masked, err_masked = \
                self.prepare_data_for_fit(restwave=restwave,\
                                          minwave=(1+self.zO3)*1200,\
                                          maxwave=(1+self.zO3)*1500)


    def fit_lya_profile_bilby(self, linemodel=lya_profile_simple):
        restwave = 1215.67
        wave_masked, flux_masked, err_masked = \
                self.prepare_data_for_fit(restwave=restwave,\
                                          minwave=(1+self.zO3)*1200,\
                                          maxwave=(1+self.zO3)*1500)

        likelihood = bilby.likelihood.GaussianLikelihood(wave_masked, flux_masked,\
                                                         linemodel, err_masked)

        wave0 = (1+self.zO3)*restwave
        fwhm0 = 500
        amp0 = np.max(flux_masked) * fwhm0 * np.sqrt(2*np.pi) / 2.355 / 3e5 * wave0
        x0 = [wave0, fwhm0, amp0, 0]
        value_extreme = np.max(np.abs(flux_masked))

        priors = dict()
        priors["center"] = bilby.core.prior.Uniform(restwave*(1+self.zO3)*(1-1/500), \
                                                    restwave*(1+self.zO3)*(1+1/500), "center")
        priors["fwhm"] = bilby.core.prior.Uniform(300, 1000, "fwhm")
        priors["flux"] = bilby.core.prior.Uniform(-amp0*10, amp0*10, "flux")
        priors["A"] = bilby.core.prior.Uniform(-value_extreme, value_extreme*2, "A")
        priors["beta"] = bilby.core.prior.Uniform(-4, 0, "beta")
        priors["wave0"] = bilby.core.prior.DeltaFunction(wave0, "wave0")

        # And run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            nlive=200,
            outdir=self.outdir,
            label=self.label,
        )
        self.lya_result = result

    def fit_civ_profile_bilby(self):
        restwave = 1549
        wave_masked, flux_masked, err_masked = \
                self.prepare_data_for_fit(restwave=restwave,\
                                          minwave=(1+self.zO3)*1500,\
                                          maxwave=(1+self.zO3)*1600)

        likelihood = bilby.likelihood.GaussianLikelihood(\
                wave_masked, flux_masked, emline_profile_simple, err_masked)

        wave0 = restwave * (1+self.zO3)
        fwhm0 = 500
        amp0 = np.max(flux_masked) * fwhm0 * np.sqrt(2*np.pi) / 2.355 / 3e5 * wave0
        x0 = [wave0, fwhm0, amp0, 0]
        value_extreme = np.max(np.abs(flux_masked))

        priors = dict()
        priors["center"] = bilby.core.prior.Uniform(restwave*(1+self.zO3)*(1-1/500), \
                                                    restwave*(1+self.zO3)*(1+1/500), "center")
        priors["fwhm"] = bilby.core.prior.Uniform(300, 1000, "fwhm")
        priors["flux"] = bilby.core.prior.Uniform(-amp0*10, amp0*10, "flux")
        priors["A"] = bilby.core.prior.Uniform(-value_extreme, value_extreme*2, "A")
        priors["beta"] = bilby.core.prior.Uniform(-4, 2, "beta")
        priors["wave0"] = bilby.core.prior.DeltaFunction(wave0, "wave0")

        # And run sampler
        result = bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler="dynesty",
            nlive=200,
            outdir=self.outdir,
            label=self.label,
        )
#        self.lya_result = result
        self.civ_result = result

    def load_lya_profile_bilby(self):
        #try:
        self.lya_result = bilby.result.read_in_result(\
                            outdir=self.outdir, label=self.label)

    def get_line_properties(self):
        # first draw a sample
        self.load_lya_profile_bilby()
        samples = self.lya_result.nested_samples
        weights = np.array(samples['weights'])

        # get the following properties:
        self.center_sample = dyfunc.resample_equal(np.array(samples['center']), weights)
        self.fwhm_sample = dyfunc.resample_equal(np.array(samples['fwhm']), weights)
        self.flux_sample = dyfunc.resample_equal(np.array(samples['flux']), weights)
        self.A_sample = dyfunc.resample_equal(np.array(samples['A']), weights)
        self.beta_sample = dyfunc.resample_equal(np.array(samples['beta']), weights)

        self.EW_sample = self.flux_sample / self.A_sample / (1+self.zO3)

        # good. We can compute things now.
        # the spectrum is in uJy, that is a weird unit.
        # lam*Flam = nu*Fnu, Flam = Fnu * c / lam**2
        speed_of_light_angstrom = 299792458 * 1e10

        self.zlya_quantile = np.percentile(self.center_sample, [0.135,2.5,16,50,84,97.5, 99.865]) / 1215.67
        self.fwhm_quantile = np.percentile(self.fwhm_sample, [0.135,2.5,16,50,84,97.5,99.865])
        self.flam_quantile = np.percentile(self.flux_sample, [0.135,2.5,16,50,84,97.5,99.865])
        self.EW_quantile = np.percentile(self.EW_sample, [0.135,2.5,16,50,84,97.5,99.865])

        # lastly, store the best fit params
        self.bestparams = self.lya_result

    def plot_contour_fit(self, plot_info):

        try:
            result = self.lya_result
        except AttributeError:
            self.load_lya_profile_bilby()
            result = self.lya_result
        result.plot_corner(save=True, filename=plot_info['corner'])

    def plot_best_fit(self, plot_info, linename):
        if linename=='lya':
            result = self.lya_result
            restwave = 1215.67
            fluxfunc = lya_profile_simple
        elif linename=='civ':
            result = self.civ_result
            restwave = 1549
            fluxfunc = emline_profile_simple

        plotdir = os.path.dirname(plot_info['bestfit'])
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)

        post_samples = result.posterior
        bestidx = np.argmax(np.array(post_samples['log_likelihood']))
        bestparams = [post_samples[l][bestidx] for l in result.parameter_labels]
        self.bestparams = np.array(bestparams)

        wave_masked, flux_masked, ivar_masked = \
                self.prepare_data_for_fit(restwave=restwave,\
                                    minwave=(1+self.zO3)*restwave*0.9,\
                                    maxwave=(1+self.zO3)*restwave*1.1)
        wave_plot = np.arange((1+self.zO3)*restwave*0.9,\
                              (1+self.zO3)*restwave*1.1, 0.1)
        wave0 = (1+self.zO3)*restwave

        plt.close('all')
        plt.plot(wave_masked, flux_masked)
        plt.plot(wave_plot, fluxfunc(wave_plot, *self.bestparams, wave0))
        plt.xlabel('Wavelength')
        plt.ylabel('Flux [uJy]')

        xlim = plot_info.pop('xlim', [wave_masked[0],wave_masked[-1]])
        plt.xlim(xlim)
        plt.title(plot_info['title']+', z=%.2f, FWHM=%.2f'%(self.zO3, bestparams[1]))
        plt.savefig(plot_info['bestfit'])

    def subtract_lya_cont(self, normalize=False, restwavegrid=None):
        self.lya_result = bilby.result.read_in_result(\
                            outdir=self.outdir, label=self.label)

        post_samples = self.lya_result.posterior
        bestidx = np.argmax(np.array(post_samples['log_likelihood']))
        bestparams = [post_samples[l][bestidx] for l in self.lya_result.parameter_labels]

        flux, A, beta = bestparams[-3:]
        wave0 = (1+self.zO3) * 1215.67
        cont = A * (self.wave/wave0)**beta
        cont[self.wave<wave0]=0
        contsub = self.flam - cont

        # prepare the data: subtract the continuum and shift to rest

        restwave = self.wave / (1+self.zO3)

        if restwavegrid is None:
            restwavegrid = np.arange(1150.17, 1300.17, 1)
#        restwavegrid = np.arange(1150, 1300, 1)
        self.restwave = restwavegrid
        self.restflux = np.interp(restwavegrid, restwave[self.mask], contsub[self.mask])
        self.restferr = np.interp(restwavegrid, restwave[self.mask], self.err_flam[self.mask])

        if normalize:
            self.restflux = self.restflux / flux
            self.restferr = self.restferr / flux

class PypeItSpec(Spectrum1D):

    def __init__(self, filename, zO3, outdir, label):
        self.filename = filename
        # read the spec
        data_coadd1d = fits.open(filename)[1].data

        self.wave = data_coadd1d['OPT_WAVE']
        self.fnu = data_coadd1d['OPT_COUNTS']
        self.ivar_fnu = data_coadd1d['OPT_COUNTS_IVAR']
        self.err_fnu = (self.ivar_fnu)**-0.5
        self.mask = data_coadd1d['OPT_MASK'] & (~np.isnan(self.fnu)) & (~np.isnan(self.err_fnu))\
                 & (~np.isinf(self.fnu)) & (~np.isinf(self.err_fnu))

        speed_of_light_angstrom = 299792458 * 1e10
        self.flam = self.fnu * speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18
        self.err_flam = self.err_fnu * (speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18)
        self.zO3 = zO3

        self.outdir = outdir
        self.label = label

class PypeItSpecStacked(Spectrum1D):
    def __init__(self, filename, zO3, outdir, label):
        self.filename = filename
        # read the spec
        data_coadd1d = fits.open(filename)[1].data

        self.wave = data_coadd1d['wave']
        self.fnu = data_coadd1d['flux']
        self.ivar_fnu = data_coadd1d['ivar']
        self.err_fnu = (self.ivar_fnu)**-0.5
        self.mask = data_coadd1d['mask']
        self.mask = self.wave>1
        #print(self.mask)

        speed_of_light_angstrom = 299792458 * 1e10
        self.flam = self.fnu * speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18
        self.err_flam = self.err_fnu * (speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18)
        self.zO3 = zO3

        self.outdir = outdir
        self.label = label

class MSAExpSpec(Spectrum1D):
    def __init__(self, filename, zO3, outdir, label):
        self.filename = filename
        # read the spec
        data = Table.read(filename)

        self.wave = data['wave'] * 1e4
        self.fnu = data['flux']
        self.err_fnu = data['err']
        self.mask = np.array([True]*len(self.wave))

        speed_of_light_angstrom = 299792458 * 1e10
        self.flam = self.fnu * speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18
        self.err_flam = self.err_fnu * (speed_of_light_angstrom / (self.wave)**2 * 1e-29 * 1e18)
        self.zO3 = zO3

        self.outdir = outdir
        self.label = label

# how to test if a line is covered?

