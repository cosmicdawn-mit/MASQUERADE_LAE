'''
Utility functions by MYue
This is just a wrapper to avoid massive notebooks
'''

import os, sys

sys.path.append('/home/minghao/Research/Projects/JWST/nirspec_msa/dependencies/PypeIt/pypeit/')
sys.path.append('/home/minghao/Research/Projects/JWST/nirspec_msa/dependencies/PypeIt-development-suite/dev_algorithms/jwst/')

# set environment variables
os.environ['CRDS_PATH'] = '/home/minghao/Research/Projects/JWST/dependencies/crds_cache/jwst_ops/'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu/'

import glob
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits

from jwst import datamodels

DO_NOT_USE = datamodels.dqflags.pixel['DO_NOT_USE']

from jwst_utils import NIRSpecSlitCalibrations, jwst_mosaic, jwst_reduce, jwst_get_slits, jwst_proc
#from jwst_targets import jwst_targets
from pypeit.metadata import PypeItMetaData
from pypeit.display import display
from pypeit.images import combineimage
from pypeit import specobjs
from pypeit.utils import inverse, fast_running_median, nan_mad_std

from pypeit.spectrographs.util import load_spectrograph
from pypeit import msgs
from pypeit import spec2dobj
from pypeit import utils
from pypeit.images.pypeitimage import PypeItImage


from pypeit.manual_extract import ManualExtractionObj
from pypeit import find_objects
from pypeit.utils import inverse, zero_not_finite

from pypeit import slittrace
from pypeit import extraction

from copy import deepcopy

from astropy.coordinates import SkyCoord
from msaexp import msa

from astropy.wcs import WCS
from astropy.nddata import Cutout2D

def find_slit_by_name(slits, name):
    for slit in slits:
        if slit.name==name:
            return slit
        
def combine_bkgs(bkg1, bkg2, scale1, scale2):
    # Subtract the image
    newimg = bkg1.image*scale1 + bkg2.image*scale2

    # Propagate the variance, if available
    if bkg1.ivar is not None or bkg2.ivar is not None:
        new_ivar = np.zeros(newimg.shape) if bkg1.ivar is None else utils.inverse(bkg1.ivar/scale1**2)
        if bkg2.ivar is not None:
            new_ivar += utils.inverse(bkg2.ivar/scale2**2)
        new_ivar = utils.inverse(new_ivar)
    else:
        new_ivar = None

    # Only create a new nimg if it's present in *both* images
    new_nimg = bkg1.nimg + bkg2.nimg if bkg1.nimg is not None and bkg2.nimg is not None \
                    else None

    # RN2
    if bkg1.rn2img is not None or bkg2.rn2img is not None:
        new_rn2 = np.zeros(newimg.shape) if bkg1.rn2img is None else bkg1.rn2img/scale1**2
        if bkg2.rn2img is not None:
            new_rn2 += bkg2.rn2img/scale2**2
    else:
        new_rn2 = None

    # Base variance (for the noise modeling, it's more important that this
    # is propagated compared to rn2)
    if bkg1.base_var is not None or bkg2.base_var is not None:
        new_base = np.zeros(newimg.shape) if bkg1.base_var is None else bkg1.base_var/scale1**2
        if bkg2.base_var is not None:
            new_base += bkg2.base_var/scale2**2
    else:
        new_base = None

    # Image scaling
    # TODO: This is bogus.  Maybe we should just set this to 1?  Either way,
    # trying to model the noise using our current approach won't be
    # mathematically correct for two subtracted images.  This will be worse
    # for more significant scaling.
    if bkg1.img_scale is not None or bkg2.img_scale is not None:
        new_img_scale = np.ones(newimg.shape) if bkg1.img_scale is None else bkg1.img_scale
        if bkg2.img_scale is not None:
            new_img_scale = (new_img_scale + bkg2.img_scale) / 2
    else:
        new_img_scale = None

    # Mask
    if bkg1.fullmask is not None or bkg2.fullmask is not None:
        new_fullmask = ImageBitMaskArray(newimg.shape) if bkg1.fullmask is None \
                            else bkg1.fullmask
        if bkg2.fullmask is not None:
            new_fullmask |= bkg2.fullmask
    else:
        new_fullmask = None

    # PYP_SPEC
    # TODO: Instead raise an error if they're not the same
    new_spec = bkg1.PYP_SPEC if bkg1.PYP_SPEC == bkg2.PYP_SPEC else None

    # units
    # TODO: Instead raise an error if they're not the same
    new_units = bkg1.units if bkg1.units == bkg2.units else None

    # Spatial flexure
    spat_flexure = bkg1.spat_flexure
    if bkg2.spat_flexure is not None and spat_flexure is not None \
            and bkg2.spat_flexure != spat_flexure:
        msgs.warn(f'Spatial flexure different for images being subtracted ({spat_flexure} '
                  f'vs. {bkg2.spat_flexure}).  Adopting {spat_flexure}.')

    # Create the new image.
    # TODO: We should instead *copy* the detector object; bkg2wise, it's
    # possible that it will be shared between multiple images.  Nominally,
    # this should be okay because the detector data is meant to be static,
    # but we should fix this.
    new_pypeitImage = PypeItImage(newimg, ivar=new_ivar, nimg=new_nimg, rn2img=new_rn2,
                                  base_var=new_base, img_scale=new_img_scale,
                                  fullmask=new_fullmask, detector=bkg1.detector,
                                  spat_flexure=spat_flexure, PYP_SPEC=new_spec,
                                  units=new_units)

    # Files
    if bkg1.files is not None and bkg2.files is not None:
        new_pypeitImage.files = bkg1.files + bkg2.files

    # Return the result using the `from_pypeitimage` instantiation method to
    # ensure the type of the output image is identical to the type of bkg1.
    # It does not matter whether it's done here or above when instantiating
    # new_pypeitimage.  Both properly propagate the `files` attribute.
    return bkg1.__class__.from_pypeitimage(new_pypeitImage)



def sub(self, other):
    """
    Subtract this PypeItImage from another.

    The following operations are performed:

        - the image data is subtracted (images must have the same shape)
        - the inverse variance (:attr:`ivar`) is propagated
        - the number of images is combined (:attr:`nimg`)
        - the RN2 (:attr:`rn2img`) is propagated
        - the base variance (:attr:`base_var`) is propagated
        - the image scaling (:attr:`img_scale`) is averaged
        - the bit mask (:attr:`fullmask`) is joined (using an or operation)
        - if it's the same for both images, the spectrograph name
          (:attr:`PYP_SPEC`) is propagated
        - if it's the same for both images, the images units (:attr:`units`)
          is propagated
        - if both images provide source file names, the file lists are
          concatenated
        - the detector from the first image (``self``) is used for the
          returned image and the detector for the ``other`` image is
          *ignored*
        - if the spatial flexure is defined for the first image, it is
          propagated regardless of the value for the 2nd image.  If it is
          also defined for the 2nd image and the flexure is different from
          the first image, a warning is issued.  If the flexure is only
          defined for the 2nd image, it is ignored.

    Args:
        other (:class:`~pypeit.images.pypeitimage.PypeItImage`):
            The image to subtract

    Returns:
        :class:`~pypeit.images.pypeitimage.PypeItImage`: The result of
        subtracting ``other`` from this image.
    """
    if not isinstance(other, PypeItImage):
        msgs.error('Image to subtract must be of type PypeItImage.')

    # Subtract the image
    newimg = self.image - other.image

    # Propagate the variance, if available
    if self.ivar is not None or other.ivar is not None:
        new_ivar = np.zeros(newimg.shape) if self.ivar is None else utils.inverse(self.ivar)
        if other.ivar is not None:
            new_ivar += utils.inverse(other.ivar)
        new_ivar = utils.inverse(new_ivar)
    else:
        new_ivar = None

    # Only create a new nimg if it's present in *both* images
    new_nimg = self.nimg + other.nimg if self.nimg is not None and other.nimg is not None \
                    else None

    # RN2
    if self.rn2img is not None or other.rn2img is not None:
        new_rn2 = np.zeros(newimg.shape) if self.rn2img is None else self.rn2img
        if other.rn2img is not None:
            new_rn2 += other.rn2img
    else:
        new_rn2 = None

    # Base variance (for the noise modeling, it's more important that this
    # is propagated compared to rn2)
    if self.base_var is not None or other.base_var is not None:
        new_base = np.zeros(newimg.shape) if self.base_var is None else self.base_var
        if other.base_var is not None:
            new_base += other.base_var
    else:
        new_base = None

    # Image scaling
    # TODO: This is bogus.  Maybe we should just set this to 1?  Either way,
    # trying to model the noise using our current approach won't be
    # mathematically correct for two subtracted images.  This will be worse
    # for more significant scaling.
    if self.img_scale is not None or other.img_scale is not None:
        new_img_scale = np.ones(newimg.shape) if self.img_scale is None else self.img_scale
        if other.img_scale is not None:
            new_img_scale = (new_img_scale + other.img_scale) / 2
    else:
        new_img_scale = None

    # Mask
    if self.fullmask is not None or other.fullmask is not None:
        new_fullmask = ImageBitMaskArray(newimg.shape) if self.fullmask is None \
                            else self.fullmask
        if other.fullmask is not None:
            new_fullmask |= other.fullmask
    else:
        new_fullmask = None

    # PYP_SPEC
    # TODO: Instead raise an error if they're not the same
    new_spec = self.PYP_SPEC if self.PYP_SPEC == other.PYP_SPEC else None

    # units
    # TODO: Instead raise an error if they're not the same
    new_units = self.units if self.units == other.units else None

    # Spatial flexure
    spat_flexure = self.spat_flexure
    if other.spat_flexure is not None and spat_flexure is not None \
            and other.spat_flexure != spat_flexure:
        msgs.warn(f'Spatial flexure different for images being subtracted ({spat_flexure} '
                  f'vs. {other.spat_flexure}).  Adopting {np.max(np.abs([spat_flexure, other.spat_flexure]))}.')

    # Create a copy of the detector, if it is defined, to be used when
    # creating the new pypeit image below
    _detector = None if self.detector is None else deepcopy(self.detector)#.copy()

    # Create the new image.
    new_pypeitImage = PypeItImage(newimg, ivar=new_ivar, nimg=new_nimg, rn2img=new_rn2,
                                  base_var=new_base, img_scale=new_img_scale,
                                  fullmask=new_fullmask, detector=_detector,
                                  spat_flexure=spat_flexure, PYP_SPEC=new_spec,
                                  units=new_units)

    # Files
    if self.files is not None and other.files is not None:
        new_pypeitImage.files = self.files + other.files

    # Return the result using the `from_pypeitimage` instantiation method to
    # ensure the type of the output image is identical to the type of self.
    # It does not matter whether it's done here or above when instantiating
    # new_pypeitimage.  Both properly propagate the `files` attribute.
    return self.__class__.from_pypeitimage(new_pypeitImage)


'''
Wrappers for reading data
'''


def load_pypeit_msa(pypeit_output_dir):
    # Some pypeit things
    spectrograph = load_spectrograph('jwst_nirspec')
    
    par = spectrograph.default_pypeit_par()
    det_container_list = [spectrograph.get_detector_par(1), spectrograph.get_detector_par(2)]

    
    pypeline = 'MultiSlit'
    par['rdx']['redux_path'] = pypeit_output_dir
    qa_dir = os.path.join(pypeit_output_dir, 'QA')
    par['rdx']['qadir'] = 'QA'
    png_dir = os.path.join(qa_dir, 'PNGs')
    if not os.path.isdir(qa_dir):
        msgs.info('Creating directory for QA output: {0}'.format(qa_dir))
        os.makedirs(qa_dir)
        if not os.path.isdir(png_dir):
            os.makedirs(png_dir)
        
        # Set some parameters for difference imaging
        #if bkg_redux:
        par['reduce']['findobj']['skip_skysub'] = True # Do not sky-subtract when object finding
        par['reduce']['extraction']['skip_optimal'] = True # Skip local_skysubtraction and profile fitting

    return spectrograph, par, det_container_list

def filename_holders(file_prefix, redux_dir, rate_suffix, file_indices):
    # glob filenames

    basenames, basenames_1, basenames_2, scifiles_1, scifiles_2 = [], [], [], [], []

    for nfile in file_indices:
        b1 = os.path.join(redux_dir, '%s_0000%d_'%(file_prefix, nfile) + 'nrs1')
        b2 = os.path.join(redux_dir, '%s_0000%d_'%(file_prefix, nfile) + 'nrs2')

        basenames_1.append(b1)
        basenames_2.append(b2)
        scifiles_1.append(os.path.join(redux_dir, b1 + '_%s.fits'%rate_suffix))
        scifiles_2.append(os.path.join(redux_dir, b2 + '_%s.fits'%rate_suffix))

    scifiles = [scifiles_1, scifiles_2]
    scifiles_all = scifiles_1 + scifiles_2

    # Output file names
    intflat_output_files_1 = []
    msa_output_files_1 = []
    cal_output_files_1 = []

    intflat_output_files_2 = []
    msa_output_files_2 = []
    cal_output_files_2 = []

    for base1, base2 in zip(basenames_1, basenames_2):

        msa_output_files_1.append(os.path.join(redux_dir, base1 + '_%s.fits'%rate_suffix))
        msa_output_files_2.append(os.path.join(redux_dir, base2 + '_%s.fits'%rate_suffix))

        intflat_output_files_1.append(os.path.join(redux_dir, base1 + '_%s_interpolatedflat.fits'%rate_suffix))
        cal_output_files_1.append(os.path.join(redux_dir, base1 + '_%s_cal.fits'%rate_suffix))
        intflat_output_files_2.append(os.path.join(redux_dir, base2 + '_%s_interpolatedflat.fits'%rate_suffix))
        cal_output_files_2.append(os.path.join(redux_dir, base2 + '_%s_cal.fits'%rate_suffix))


    holder = [[scifiles_1, intflat_output_files_1, msa_output_files_1, cal_output_files_1],\
              [scifiles_2, intflat_output_files_2, msa_output_files_2, cal_output_files_2]]

    return holder

def get_ypos(slit_dm, spec2dfile):
    center = fits.open(spec2dfile)[-2].data['center']
    
    # get the slit-frame y position
    transformer = slit_dm.meta.wcs.get_transform('world', 'slit_frame')
    
    transformer1 = slit_dm.meta.wcs.get_transform('world', 'detector')
    transformer2 = slit_dm.meta.wcs.get_transform('detector', 'slit_frame')

    ra_src, dec_src = slit_dm.source_ra, slit_dm.source_dec
    result = transformer(ra_src, dec_src, 1)

    dx, dy, w = result
    
    detx, dety = transformer1(ra_src, dec_src, 1)
    dxtest, dytest, _ = transformer2(detx, dety+1)
    #print(detx, dety)
    factor = dytest - dy
    dy_pix = dy / factor
    ypos = center + dy_pix
    
    return np.median(ypos)

# a super wrapper

class MSAreducer(object):
    '''
    A wrapper for data reading and reduction for MSA
    '''
    def __init__(self, filelist, pypeit_output_dir):
        '''
        Initializing the reducer by feeding the list of filenames it needs.
        Only feed nirs1 files; I will make the reducer find the nrs2files by itself.
        
        Input:
            filelist: list of str
                The list of nrs1 rate files to be reduced.
                The strings should be the full path.

            pypeit_output_dir: str
                the directory to hold pypeit reduction
        '''
#        print('test')
        self.pypeit_output_dir = pypeit_output_dir

        self.basenames = [os.path.basename(f)[:25] for f in filelist]

        self.scifiles_1 = [f[:-5]+'_nsclean.fits' for f in filelist]
        self.intflat_output_files_1 = [f[:-5]+'_interpolatedflat.fits' for f in filelist]
        self.cal_output_files_1 = [f[:-5]+'_cal.fits' for f in filelist]


        self.scifiles_2 = [f.replace('_nrs1_', '_nrs2_') for f in self.scifiles_1]
        self.intflat_output_files_2 = [f.replace('_nrs1_', '_nrs2_') for f in self.intflat_output_files_1]
        self.cal_output_files_2 = [f.replace('_nrs1_', '_nrs2_') for f in self.cal_output_files_1]

        self.nexp = len(filelist)
        self.bkg_redux = True

        self.initiate_spectrograph()
        self.load_data()
        self.get_slit_sources()

        # initial soem holders
        self.Single2dSpecHolder = {}
        self.Coadd2dSpecHolder = {}
        self.Extract1dSpecHolder = {}

    def initiate_spectrograph(self, **kwargs):
        pypeit_output_dir = self.pypeit_output_dir

        spectrograph = load_spectrograph('jwst_nirspec')
        
        par = spectrograph.default_pypeit_par()
        det_container_list = [spectrograph.get_detector_par(1), spectrograph.get_detector_par(2)]

        pypeline = 'MultiSlit'
        par['rdx']['redux_path'] = pypeit_output_dir
        qa_dir = os.path.join(pypeit_output_dir, 'QA')
        par['rdx']['qadir'] = 'QA'
        png_dir = os.path.join(qa_dir, 'PNGs')
        if not os.path.isdir(qa_dir):
            msgs.info('Creating directory for QA output: {0}'.format(qa_dir))
            os.makedirs(qa_dir)
            if not os.path.isdir(png_dir):
                os.makedirs(png_dir)
            
        # Set some parameters for difference imaging
        #if bkg_redux:
        par['reduce']['findobj']['skip_skysub'] = True # Do not sky-subtract when object finding
        par['reduce']['extraction']['skip_optimal'] = True # Skip local_skysubtraction and profile fitting


        self.scipath = os.path.join(self.pypeit_output_dir, 'Science')

        self.spectrograph = spectrograph
        self.par = par
        self.det_container_list = det_container_list

        return spectrograph, par, det_container_list

    def load_data(self):

        det_container_list = [self.spectrograph.get_detector_par(1), self.spectrograph.get_detector_par(2)]
        self.det_container_list = det_container_list

        # Read in multi exposure calwebb outputs
        msa_multi_list_1 = []
        intflat_multi_list_1 = []
        final_multi_list_1 = []
        msa_multi_list_2 = []
        intflat_multi_list_2 = []
        final_multi_list_2 = []

        nexp = len(self.scifiles_1)
        ndetectors = 2

        # TODO: This probably isn't correct.  I.e., need to know offsets and slit
        # position angle.

        # saving the offsets
        dither_offsets = np.zeros((ndetectors,nexp), dtype=float)
        for iexp in range(nexp):
            with fits.open(self.scifiles_1[iexp]) as hdu:
                dither_offsets[0,iexp] = hdu[0].header['YOFFSET']

        for idet in range(1,ndetectors):
            dither_offsets[idet] = dither_offsets[0]
        dither_offsets_pixels = dither_offsets.copy()

        for idet in range(ndetectors):
            dither_offsets_pixels[idet] /= det_container_list[idet].platescale

        # NOTE: Sign convention requires this calculation of the offset
        dither_offsets_pixels = dither_offsets_pixels[:,0,None] - dither_offsets_pixels
        #print(dither_offsets_pixels)

        # Create arrays to hold JWST spec2, but only load the files when they're needed
        msa_data = np.empty((ndetectors, nexp), dtype=object)
        flat_data = np.empty((ndetectors, nexp), dtype=object)
        cal_data = np.empty((ndetectors, nexp), dtype=object)


        for iexp in range(nexp):
            # Open some JWST data models
            #e2d_multi_list_1.append(datamodels.open(e2d_output_files_1[iexp]))
            msa_data[0, iexp] = datamodels.open(self.scifiles_1[iexp])
            flat_data[0, iexp] = datamodels.open(self.intflat_output_files_1[iexp])
            cal_data[0, iexp] = datamodels.open(self.cal_output_files_1[iexp])

            msa_data[1, iexp] = datamodels.open(self.scifiles_2[iexp])
            flat_data[1, iexp] = datamodels.open(self.intflat_output_files_2[iexp])
            cal_data[1, iexp] = datamodels.open(self.cal_output_files_2[iexp])


        self.fitstbl_1 = PypeItMetaData(self.spectrograph, par=self.par,\
                                        files=self.scifiles_1, strict=True)
        self.fitstbl_2 = PypeItMetaData(self.spectrograph, par=self.par,\
                                        files=self.scifiles_2, strict=True)

        self.msa_data = msa_data
        self.flat_data = flat_data
        self.cal_data = cal_data
        self.dither_offsets_pixels = dither_offsets_pixels

        return msa_data, flat_data, cal_data, dither_offsets_pixels

    def get_slit_sources(self):
        slit_names_1 = [slit.name for slit in self.cal_data[0,0].slits]
        slit_names_2 = [slit.name for slit in self.cal_data[1,0].slits]
        slit_names_tot = np.hstack([slit_names_1, slit_names_2])
        source_names_1 = [slit.source_name for slit in self.cal_data[0,0].slits]
        source_names_2 = [slit.source_name for slit in self.cal_data[1,0].slits]
        source_names_tot = np.hstack([source_names_1, source_names_2])

        # Find the unique slit names and the unique sources aligned with those slits
        slit_names_uni, uni_indx = np.unique(slit_names_tot, return_index=True)
        source_names_uni = source_names_tot[uni_indx]
        slit_sources_uni = [(slit, source) for slit, source in zip(slit_names_uni, source_names_uni)]

        self.slits_sources = slit_sources_uni

    def find_slit_by_islit(self, islit):
        slits_1 = self.cal_data[0,0].slits
        slits_2 = self.cal_data[1,0].slits

        for slit in slits_1:
            if slit.name==islit:
                return slit

        for slit in slits_2:
            if slit.name==islit:
                return slit

        return None

    def find_islit_by_isource(self, isource):
        for isl, iso in self.slits_sources:
            if iso==isource:
                break

        return isl

    def reduce_slit(self, islit, isource, ibkg_list=None):

        iexp_ref = 0
        kludge_err = 1

        CalibrationsNRS1 = NIRSpecSlitCalibrations(self.det_container_list[0],
                                               self.cal_data[0, iexp_ref],
                                                self.flat_data[0, iexp_ref],
                                               islit, f070_f100_rescale=False)
        CalibrationsNRS2 = NIRSpecSlitCalibrations(self.det_container_list[1],
                                                   self.cal_data[1, iexp_ref],
                                                   self.flat_data[1, iexp_ref],
                                                   islit, f070_f100_rescale=False)

        msa_data = self.msa_data
        cal_data = self.cal_data
        nexp = self.nexp

        spec2d_file_list = []

        for iexp in range(self.nexp):

            # Container for all the Spec2DObj, different spec2dobj and specobjs for each slit
            all_spec2d = spec2dobj.AllSpec2DObj()
            all_spec2d['meta']['bkg_redux'] = self.bkg_redux
            all_spec2d['meta']['find_negative'] = self.bkg_redux
            # Container for the specobjs
            all_specobjs = specobjs.SpecObjs()

            #ibkg = bkg_indices[iexp]
            # Create the image mosaic
            sciImg, slits, waveimg, tilts, ndet = \
                jwst_mosaic(msa_data[:, iexp], [CalibrationsNRS1, CalibrationsNRS2], kludge_err=kludge_err,
                noise_floor=self.par['scienceframe']['process']['noise_floor'])

            # Do bkg subtraction using dithering

            bkgimglist = []

            if ibkg_list==None:
                ibkg_all = list(range(nexp))
                ibkg_all.remove(iexp)
            else:
                ibkg_all = ibkg_list[iexp]

            bkgImg1, _, _, _, _ = jwst_mosaic(msa_data[:, ibkg_all[0]], [CalibrationsNRS1, CalibrationsNRS2],
                                                  kludge_err=kludge_err,
                                              noise_floor=self.par['scienceframe']['process']['noise_floor'])
            #bkgImg1 = add_cr_mask(bkgImg1)

            if len(ibkg_all)>1:
                bkgImg2, _, _, _, _ = jwst_mosaic(msa_data[:, ibkg_all[1]], [CalibrationsNRS1, CalibrationsNRS2],
                                                  kludge_err=kludge_err,
                                              noise_floor=self.par['scienceframe']['process']['noise_floor'])
                bkgImg = combine_bkgs(bkgImg1, bkgImg2, 0.5, 0.5)
            else:
                bkgImg = bkgImg1

            sciImg = sub(sciImg, bkgImg)

            # TODO the parset label here may change in Pypeit to bkgframe
            # MYue: this does not work for me. Back to the old version
            #combineImage = combineimage.CombineImage(bkgImg_list, spectrograph, par['scienceframe']['process'])
            #bkgImg = combineImage.run(ignore_saturation=True)
            #sciImg = sciImg.sub(bkgImg)


            # Run the reduction
            all_spec2d[sciImg.detector.name], tmp_sobjs = \
                        jwst_reduce(sciImg, slits, waveimg, tilts,\
                                    self.spectrograph, self.par, show=False,\
                                    find_negative=self.bkg_redux, bkg_redux=self.bkg_redux,
                                    clear_ginga=False, show_peaks=False, show_skysub_fit=False,
                                    basename=self.basenames[iexp])

            # Hold em
            if tmp_sobjs.nobj > 0:
                all_specobjs.add_sobj(tmp_sobjs)

            # THE FOLLOWING MIMICS THE CODE IN pypeit.save_exposure()
            base_suffix = 'source_{:s}'.format(isource) if isource is not None else 'slit_{:s}'.format(islit)
            basename = '{:s}_{:s}'.format(self.basenames[iexp], base_suffix)

            # TODO Populate the header with metadata relevant to this source?

            # Write out specobjs
            # Build header for spec2d
            head2d = fits.getheader(self.scifiles_1[iexp])
            subheader = self.spectrograph.subheader_for_spec(self.fitstbl_1[iexp], head2d, allow_missing=False)
            # Overload the target name with the source name
            subheader['target'] = isource

            if all_specobjs.nobj > 0:
                outfile1d = os.path.join(self.scipath, 'spec1d_{:s}.fits'.format(basename))
                all_specobjs.write_to_fits(subheader, outfile1d)

            # Info
            outfiletxt = os.path.join(self.scipath, 'spec1d_{:s}.txt'.format(basename))
            all_specobjs.write_info(outfiletxt, self.spectrograph.pypeline)

            # Build header for spec2d
            outfile2d = os.path.join(self.scipath, 'spec2d_{:s}.fits'.format(basename))
            # TODO For the moment hack so that we can write this out
            pri_hdr = all_spec2d.build_primary_hdr(head2d, self.spectrograph, subheader=subheader,
                                                   redux_path=None, calib_dir=None)
            # Write spec2d
            os.system('rm %s'%outfile2d)
            all_spec2d.write_to_fits(outfile2d, pri_hdr=pri_hdr, overwrite=True)
            spec2d_file_list.append(outfile2d)

        self.Single2dSpecHolder[isource] = deepcopy(spec2d_file_list)

    def coadd2d_source(self, islit, isource, outdir=None, **kwargs):
        '''
        producing the coadded 2d spectrum
        '''

        # find filenames
        sci_dir = self.scipath
        filenames_all = self.Single2dSpecHolder[isource]

        if 'stack_indices' in kwargs.keys():
            stack_indices = kwargs['stack_indices']
            filenames = [filenames_all[i] for i in stack_indices]
        else:
            filenames = filenames_all

        # get the detector name
        header = fits.getheader(filenames[0], 1)

        if 'DET01' in header['EXTNAME']:
            detnum = '1'
        elif 'MSC01' in header['EXTNAME']:
            detnum = '(1,2)'

        manual_str = ''

        find_trim_edge = kwargs.pop('find_trim_edge', '10,10')
        find_min_max = kwargs.pop('find_min_max', '100,800')

        if 'offsets' in kwargs.keys():
            offsets = kwargs['offsets']

        elif len(filenames)==3:
            offsets = [0.0 , -5.28973998,  5.29020131]

        elif len(filenames)==6:
            offsets = [0.0 , -5.28973998,  5.29020131] * 2

        else:
            raise ValueError('No offset found in kwargs')

        assert len(filenames)==len(offsets)

        offset_string = "offsets = "+ ', '.join(['%.7f']*len(offsets)) % tuple(offsets)

        params_string = \
'''
[rdx]
  spectrograph = jwst_nirspec
  detnum = {}
[reduce]
  [[findobj]]
   skip_skysub=True
   snr_thresh = 3.0
   trace_npoly = 2
   find_fwhm = 2.0
   maxnumber_sci = 10
   find_trim_edge = {}
   find_min_max = {}
  [[skysub]] 
    no_local_sky = True
  [[extraction]] 
    use_user_fwhm=False 
    skip_optimal=False
    sn_gauss = 2.8
[coadd2d]
   use_slits4wvgrid = True
   wave_method = iref
   {}
   weights = uniform
   {}
'''.format(detnum, find_trim_edge, find_min_max, offset_string, manual_str)

        filename_string = '\n'.join(['%s']*len(filenames)) % tuple(filenames)

        file_string = \
'''
# Read in the data
spec2d read
path {}
filename
{}
spec2d end
'''.format(os.path.abspath(sci_dir), filename_string)

        string_to_write = params_string+file_string

        cwd = os.getcwd()
        scriptname = cwd+'/coadd_scripts/coadd2d_%s'%isource

        if not os.path.isdir(os.path.dirname(scriptname)):
            os.mkdir(os.path.dirname(scriptname))

        with open(scriptname, 'w') as f:
            f.write(string_to_write)

        coadd2d_command = kwargs.pop('coadd2d_command', 'pypeit_coadd_2dspec')
        os.system('%s %s'%(coadd2d_command, scriptname))
#        os.system('rm *.par')
        if outdir is not None:
            print('copying: ')

            scifile1name = os.path.basename(self.scifiles_1[0])[:-5]
            scifile2name = os.path.basename(self.scifiles_1[-1])[:-5]

#            spec2d_to_be_copied = glob.glob('Science_coadd/spec2d_%s_*%s.fits'%(self.basenames[0], isource))
#            spec1d_to_be_copied = glob.glob('Science_coadd/spec1d_%s_*%s.fits'%(self.basenames[0], isource))
            spec2d_to_be_copied = f'Science_coadd/spec2d_{scifile1name}-{scifile2name}-{isource}.fits'
            spec1d_to_be_copied = f'Science_coadd/spec1d_{scifile1name}-{scifile2name}-{isource}.fits'

            for f in [spec2d_to_be_copied, spec1d_to_be_copied]:
                os.system('cp %s %s'%(f, outdir))

            new_filenames = outdir+'/'+os.path.basename(spec2d_to_be_copied)
            print(new_filenames)
#            assert len(new_filenames)==1
            assert os.path.exists(new_filenames)
            self.Coadd2dSpecHolder[isource] = new_filenames

    def manual_extraction(self, islit, isource, manual_info=None, **kwargs):
        spectrograph = self.spectrograph
        bkg_redux = self.bkg_redux

        stacked2dfile = self.Coadd2dSpecHolder[isource]
        newpar = deepcopy(self.par)
        detname = fits.getheader(stacked2dfile, 1)['EXTNAME'].split('-')[0]
        spec2d = spec2dobj.Spec2DObj.from_file(stacked2dfile, detname, chk_version=False)

        if manual_info is None:
            slit_dm = self.find_slit_by_islit(islit)
            ypos = get_ypos(slit_dm, stacked2dfile)
            manual_info = {'ypos': ypos, 'xpos': 400.0, 'fwhm':3.0,'boxcar_rad':3.0}

        newpar['reduce']['extraction']['skip_optimal'] = False
        newpar['reduce']['findobj']['skip_skysub'] = True

        newpar['reduce']['findobj']['snr_thresh'] = 10000000
        newpar['reduce']['skysub']['no_local_sky'] = True
        manual_obj = ManualExtractionObj(frame=stacked2dfile, \
                            spat=np.array([manual_info['ypos']]), spec=np.array([manual_info['xpos']]),\
                            detname=np.array([spec2d.detector.name]),\
                            fwhm=np.array([manual_info['fwhm']]), neg=np.array([False]),\
                            boxcar_rad=np.array([manual_info['boxcar_rad']]))

        sciImg = PypeItImage(image=zero_not_finite(spec2d.sciimg), ivar=zero_not_finite(spec2d.ivarraw),
                             base_var=zero_not_finite(np.zeros(spec2d.sciimg.shape)),
                             img_scale=zero_not_finite(np.ones(spec2d.sciimg.shape)),
                             rn2img=zero_not_finite(np.zeros(spec2d.sciimg.shape)),
                             detector=spec2d.detector, bpm=np.array(spec2d.bpmmask['mask'], dtype=bool))

        shape = spec2d.sciimg.shape
        slit_left_tot, slit_righ_tot = jwst_get_slits(spec2d.bpmmask['mask'])

        slits = slittrace.SlitTraceSet(slit_left_tot, slit_righ_tot, 'MultiSlit', detname=spec2d.detector.name,
                                       nspat=int(shape[1]),PYP_SPEC='jwst_nirspec')

        objFind = find_objects.FindObjects.get_instance(sciImg, slits, spectrograph, newpar,
                                                        'science_coadd2d', tilts=spec2d.tilts,
                                                        bkg_redux=True, manual=manual_obj,
                                                        find_negative=False, basename=None,
                                                        clear_ginga=True, show=False)
                                                       #initial_skymask=None)
        
        global_sky0, sobjs_obj = objFind.run(show_peaks=False, show_skysub_fit=False)
        final_global_sky=global_sky0

        exTract = extraction.Extract.get_instance(sciImg, slits, sobjs_obj, spectrograph, newpar,
                                                  'science_coadd2d', global_sky=final_global_sky, tilts=spec2d.tilts,
                                                  waveimg=spec2d.waveimg, bkg_redux=bkg_redux,
                                                  basename=None, show=False)

        skymodel,bkg_redux_skymodel, objmodel, ivarmodel, outmask, sobjs, _, _, slits_out = exTract.run()
        for sobj in sobjs:
            sobj.DETECTOR = sciImg.detector

        # save the objects
        all_specobjs = specobjs.SpecObjs()
        all_specobjs.add_sobj(sobjs) 

        head2d = fits.getheader(stacked2dfile)
        subheader = spectrograph.subheader_for_spec(self.fitstbl_1[0], head2d, allow_missing=False)
        # Overload the target name with the source name
        subheader['target'] = isource

        outfile1d = stacked2dfile.replace('spec2d', 'manual1d')

        sobjs.write_to_fits(subheader, outfile1d)

        return outfile1d


    # plotter class
    def saveinfo(self, tbl_info):
        islitlist = []
        isourcelist = []
        ralist = []
        declist = []
        zlist = []
        maglist = []
        spec2dlist = []
        spec1dlist = []
        manual1dlist = []

        for ii, (islit, isource) in enumerate(self.slits_sources):
            islitlist.append(islit)
            isourcelist.append(isource)

            #slit_dm = find_slit_by_name(cal_data[0,0].slits, islit)
            slit_dm = self.find_slit_by_islit(islit)
            ra_src, dec_src = slit_dm.source_ra, slit_dm.source_dec

            # find the object
            objid = int(isource.split('_')[-1])
            infodict = get_infodict(objid, tbl_info)

            ralist.append(ra_src)
            declist.append(dec_src)
            zlist.append(infodict['z'])
            maglist.append(infodict['mag'])


            # get positions
            spec2dname = self.Coadd2dSpecHolder[isource]
            spec1dname = spec2dname.replace('spec2d', 'spec1d')
            manual1dname = spec2dname.replace('spec2d', 'manual1d')

            spec2dlist.append(spec2dname)
            spec1dlist.append(spec1dname)
            manual1dlist.append(manual1dname)
        
        self.tbl_extraction_info = Table({
            'islit': islitlist,\
            'isource': isourcelist,\
            'RA': ralist,\
            'Dec': declist,\
            'redshift': zlist,\
            'F115W': maglist,\
            'spec2dlist': spec2dlist,\
            'spec1dlist': spec1dlist,\
            'manual1dlist': manual1dlist,\
        })


# this is a small function to get the info of the object from EIGER catalog
def get_infodict(objid, tbl_info):
    try:
        tbl_thisobj = tbl_info[tbl_info['NUMBER']==objid]
        z = tbl_thisobj['redshift'][0]
        mag = tbl_thisobj['MAG_AUTO_F115W_apcor'][0]
    
        infodict = {'ID':objid, 'z': z, 'mag': mag}
    except:
        z = 0
        infodict = {'ID':objid, 'z': 0, 'mag': -1}
        
    return infodict

class MSAPlotter(object):
    def __init__(self, ra, dec, isource, redshift, mag, \
                 imagefile, spec2dfile, spec1dfile, msafile, direct=None, zqso=None):
        if direct is None:
            self.direct = os.getcwd()
        else:
            self.direct = direct
            
        self.isource = isource
        self.coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        self.redshift = redshift
        self.mag = mag
        
        self.imagefile = imagefile
        self.spec2dfile = spec2dfile
        self.spec1dfile = spec1dfile
        self.msafile = msafile
        self.zqso = zqso

    def initialize_axes(self, figsize=[11,4]):
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        widths = [2,7]
        heights = [1, 2]
        gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights)

        title = '%s, z=%.2f, F115W=%.2f'%(self.isource, self.redshift, self.mag)
        if 'manual' in self.spec1dfile:
            title += ' (manual)'
        fig.suptitle(title)

        self.fig = fig
        self.ax_spec2d = fig.add_subplot(gs[0,:])
        self.ax_image = fig.add_subplot(gs[1,0])
        self.ax_spec1d = fig.add_subplot(gs[1,1])

    def load_image(self):
        self.imagehdu = fits.open(self.imagefile)[1]

    def load_spec2d(self):
        data_coadd2d = fits.open(self.spec2dfile)

        sci2d =  data_coadd2d[1].data.T
        scale2d = np.percentile(sci2d, [5, 95])
        wav2d = data_coadd2d[8].data.T/1e4

        ymax, xmax = wav2d.shape

        center_xlist = np.arange(xmax)
        center_ylist = np.ones(xmax) * 15

        extent2d = [0, xmax, 0, ymax]

        self.spec2dinfo = dict(sci2d=sci2d, wav2d=wav2d,\
                               center_xlist=center_xlist,\
                               center_ylist=center_ylist, \
                               extent2d=extent2d)

    def load_spec1d(self):
        data_coadd1d = fits.open(self.spec1dfile)[1].data

        wave = data_coadd1d['OPT_WAVE']/1e4
        flux = data_coadd1d['OPT_COUNTS']
        ferr = data_coadd1d['OPT_COUNTS_IVAR']**(-0.5)
        ytrace = data_coadd1d['TRACE_SPAT']

        #scale1d = np.nanpercentile(spec1d['flux'], [1, 99])
        self.spec1dinfo = {'wave': wave, 'flux': flux, 'ferr': ferr,\
                           'ytrace': ytrace}

    def plot_image(self, **kwargs):
        if not hasattr(self, 'imagehdu'):
            self.load_image()

        percentile_img = kwargs.pop('percentile_img', [1,99])
        scale_img = np.percentile(self.imagehdu.data[50:150,50:150], percentile_img)
        self.ax_image.imshow(self.imagehdu.data, origin='lower',\
                             vmin=scale_img[0], vmax=scale_img[1])
        self.ax_image.get_yaxis().set_visible(False)
        self.ax_image.get_xaxis().set_visible(False)

        regions = msa.regions_from_metafile(self.msafile)

        wcs = WCS(self.imagehdu.header)
        for r in regions:
            if str(r.meta['source_id']) in self.isource:
                regioncoords = r.xy[0]

                regionx = []
                regiony = []

                for index in range(len(regioncoords)):
                    thiscoord = SkyCoord(ra=regioncoords[index,0],\
                                         dec=regioncoords[index,1], unit='deg')
                    thisx, thisy = wcs.world_to_pixel(thiscoord)
                    regionx.append(thisx)
                    regiony.append(thisy)

                for index in range(len(regionx)):
                    index2 = (index + 1)%len(regionx)

                    self.ax_image.plot([regionx[index], regionx[index2]],\
                                      [regiony[index], regiony[index2]],\
                                      'w-')

    def plot_spec1d(self, **kwargs):

        self.ax_spec1d.cla()

        if not hasattr(self, 'spec1info'):
            self.load_spec1d()

        self.ax_spec1d.step(self.spec1dinfo['wave'],\
                            self.spec1dinfo['flux'],\
                            label='Flux')
        self.ax_spec1d.step(self.spec1dinfo['wave'],\
                            self.spec1dinfo['ferr'],\
                            label='Error')

        percentile = kwargs.pop('percentile', [1,99])
        scale1d = np.nanpercentile(self.spec1dinfo['flux'], percentile)
        self.ax_spec1d.set_ylim(scale1d)
        self.ax_spec1d.set_xlim([0.7,1.27])

        #self.ax_spec1d.set_title('Extracted 1D')
        self.ax_spec1d.set_ylabel(r'Flux [$\mu$Jy]')
        self.ax_spec1d.set_xlabel('Wavelength [micron]')
        self.ax_spec1d.legend()

        if self.redshift>0:
            # add lines
            linewaves = np.array([1215.67, 1240.81, 1549.48, 1640.4])*(1+self.redshift)/1e4
            linenames = np.array(['Lya', 'NV', 'CIV', 'HeII'])
            liney = 0.1 * scale1d[0] + 0.9 * scale1d[1]

            for linewave, linename in zip(linewaves, linenames):
                self.ax_spec1d.text(linewave+0.001, liney, linename, color='r')
                self.ax_spec1d.plot(linewave, liney, 'r|')

            qsolya_wave = 1215.67*(1+self.zqso)/1e4

            self.ax_spec1d.text(qsolya_wave+0.001, liney, 'Lya (QSO)', color='y')
            self.ax_spec1d.plot(qsolya_wave, liney, 'y|')

        if 'estimated_flux' in kwargs.keys():
            eflux = kwargs['estimated_flux']
            mflux = np.median(self.spec1dinfo['flux'])
            self.ax_spec1d.plot([0.7, 1.27], [eflux, eflux], 'r--')
        self.ax_spec1d.plot([0.7, 1.27], [mflux, mflux], 'b--')
        self.ax_spec1d.plot([0.7, 1.27], [0, 0], 'k--', alpha=0.5)

    def plot_spec2d(self, **kwargs):
        if not hasattr(self, 'fig'):
            self.initialize_axes()
        else:
            self.ax_spec2d.cla()

        if not hasattr(self, 'spec2dinfo'):
            self.load_spec2d()
        if not hasattr(self, 'spec1dinfo'):
            self.load_spec1d()

        sci2d = self.spec2dinfo['sci2d']
        #extent2d = self.spec2dinfo['extent2d']
        ymax, xmax = sci2d.shape

        percentile = kwargs.pop('percentile_2d', [5,95])
        scale2d = np.nanpercentile(sci2d, percentile)
        vmin, vmax = scale2d
        self.ax_spec2d.imshow(sci2d, vmin=vmin, vmax=vmax, origin='lower',\
                              #extent=extent2d, aspect=dx_dy,\
                              cmap='gray', aspect='equal')
        self.ax_spec2d.get_yaxis().set_visible(False)
        self.ax_spec2d.get_xaxis().set_visible(False)

        self.ax_spec2d.plot(self.spec1dinfo['ytrace'], 'r--', alpha=0.2, lw=1)

        if self.redshift>0:
            # add lines
            linewaves = np.array([1215.67, 1240.81, 1549.48, 1640.4])*(1+self.redshift)/1e4
            linenames = np.array(['Lya', 'NV', 'CIV', 'HeII'])
            linecoords = self.wave_to_xcoord(linewaves, np.median(self.spec1dinfo['ytrace']))

            qsolya_wave = 1215.67*(1+self.zqso)/1e4
            qsolya_xcoord = self.wave_to_xcoord(1216*(1+self.zqso)/1e4,\
                                np.median(self.spec1dinfo['ytrace']))

            self.ax_spec2d.plot([qsolya_xcoord, qsolya_xcoord], [0,5], 'yellow')

            for linewave, linename, linecoord in zip(linewaves, linenames, linecoords):
                self.ax_spec2d.plot([linecoord, linecoord], [0,5], color='red')
                self.ax_spec2d.text(linecoord+3, 1, linename, color='red')

            qsolya_wave = 1216*(1+self.zqso)/1e4

    def plot(self, **kwargs):
        if not hasattr(self, 'fig'):
            self.initialize_axes()
        self.plot_spec1d(**kwargs)
        self.plot_spec2d(**kwargs)
        self.plot_image(**kwargs)

        if 'output' in kwargs.keys():
            self.fig.savefig(kwargs['output'])

    def wave_to_xcoord(self, wave, ypos=25):
        if not hasattr(self, 'spec2dinfo'):
            self.load_spec2d()

        wav2d = self.spec2dinfo['wav2d']
        '''
        Convert the wavelength to x-coordinate based on the 2D wavelength array
        '''
        # find the middle lane
        wav1d = wav2d[int(ypos)]

        xcoord = np.arange(len(wav1d))
        xinterp = np.interp(wave, wav1d, xcoord)

        return xinterp

    def make_cutout(self, masterimagename, output, ext=1, overwrite=False, size=[100,100]):
        if self.imagefile is not None and os.path.exists(self.imagefile) and overwrite==False:
            print('The image cutout have already been generated.')
            self.load_image()
        else:
            # Load the image and the WCS
            hdu = fits.open(masterimagename)[ext]
            wcs = WCS(hdu.header)
            position = wcs.world_to_pixel(self.coord)
            print(position)

            # Make the cutout, including the WCS
            cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)

            # Put the cutout image in the FITS HDU
            hdu.data = cutout.data

            # Update the FITS header with the cutout WCS
            hdu.header.update(cutout.wcs.to_header())

            # Write the cutout to a new FITS file
            hdu.writeto(output, overwrite=True)
            self.imagefile = output

            self.load_image()

