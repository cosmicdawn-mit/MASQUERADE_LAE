#!/usr/bin/env python

# NIRSpec preprocessing steps up to slitlet extractions

import sys

# Quiet!
import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['CRDS_CONTEXT'] = os.environ['CRDS_CTX'] = 'jwst_1225.pmap'
#os.environ['CRDS_PATH']='/Users/jmatthee/Documents/storage/MASQUERADE/j0100data/crds_cache'
#os.environ['CRDS_SERVER_URL']='https://jwst-crds.stsci.edu'

os.environ["CRDS_PATH"] = "/Users/myue/Research/Projects/JWST/dependencies/crds_cache/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"



# REDUCTION_VERSION = '-v3'

from grizli import jwst_level1
from msaexp import pipeline
import mastquery.jwst
import mastquery.utils

import os
import glob
import yaml
import warnings
import time

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import astropy.time

import grizli
from grizli import utils, jwst_utils
jwst_utils.set_quiet_logging()
utils.set_warnings()

import astropy.io.fits as pyfits
from jwst.datamodels import SlitModel

import msaexp
from msaexp import pipeline
try:
    from msaexp import pipeline_extended
    HAS_EXTENDED_PIPELINE = True
except:
    HAS_EXTENDED_PIPELINE = False

import jwst
import time

NIRSPEC_HOME = '/Users/jmatthee/Documents/storage/NIRspec/'
HOME = os.getcwd()


def test():

    go = """
    preprocess_nirspec_file(
        rate_file='jw01286005001_09101_00001_nrs1_rate.fits'
        root='jades-gds05-v4'
        as_fixed=False
        rename_f070=True
        context='jwst_1293.pmap'
    )
    """

def run_one():
    """
    Run a file with status = 0
    """
    import os
    import time

    from grizli.aws import db
    from grizli import utils


    print(f'============  Preprocess NIRSpec  ==============')
    print(f'========= {time.ctime()} ==========')

    #file_prefix = rate_file.split('_rate')[0]
    # key = row['root'][0] + '-' + file_prefix

    WORKPATH = HOME

    # with open(os.path.join(HOME, 'nirspec_prep_history.txt'),'a') as fp:
    #     fp.write(f"{time.ctime()} {rate_file}\n")


    LIST=glob.glob(HOME+'/*nrs*_rate.fits') ####
    for q in LIST:
        status = preprocess_nirspec_file(q,root='masq-j0100', clean=False, extend_wavelengths=False)

    return 


def new_filename(rate_file='jw01286005001_03101_00002_nrs2_rate.fits', c='b'):
    """
    """
    spl = rate_file.split('_')
    spl[2] = c + spl[2][1:]
    return '_'.join(spl)


def preprocess_nirspec_file(rate_file='jw01286005001_03101_00002_nrs2_rate.fits', root='jades-gds05-v3', as_fixed=False, rename_f070=False, context='jwst_1225.pmap', clean=True, extend_wavelengths=True, undo_flat=True, by_source=False, **kwargs):
    """
    Run preprocessing calibrations for a single NIRSpec exposure
    """
    from grizli import jwst_level1

    os.environ['CRDS_CONTEXT'] = os.environ['CRDS_CTX'] = context
    jwst_utils.set_crds_context()

    # print(rate_file, root)

    outroot = root

    if extend_wavelengths:
        rename_f070 = False

    file_prefix = rate_file.split('_rate')[0]
    # key = root + '-' + file_prefix
    key = f'{root}-{file_prefix}-{rename_f070}'

    WORKPATH = os.path.join(HOME, key)

    OUTPUT_PATH = f'{NIRSPEC_HOME}/{outroot}'

    if not os.path.exists(WORKPATH):
        os.makedirs(WORKPATH)

    os.chdir(WORKPATH)

    _ORIG_LOGFILE = utils.LOGFILE
    _NEW_LOGFILE = os.path.join(WORKPATH, file_prefix + '_rate.log.txt')
    utils.LOGFILE = _NEW_LOGFILE

    msg = f"""# {rate_file} {root}
  jwst version = {jwst.__version__}
grizli version = {grizli.__version__}
msaexp version = {msaexp.__version__}
    """
    utils.log_comment(utils.LOGFILE, msg, verbose=True)

    # Download to working directory
    mastquery.utils.download_from_mast([rate_file], overwrite=False)

    os.system(f'aws s3 cp s3://grizli-v2/reprocess_rate/{rate_file} .')

    if not os.path.exists(rate_file):
        msg = f"Failed to download {rate_file}"
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        return 3

    if not as_fixed:
        with pyfits.open(rate_file) as im:
            if 'MSAMETFL' in im[0].header:
                msametf = im[0].header['MSAMETFL']
                mastquery.utils.download_from_mast([msametf], overwrite=False)
                
                msa = msaexp.msa.MSAMetafile(msametf)
                msa.merge_source_ids()
                msa.write(prefix='', overwrite=True)
                
            else:
                msametf = None
    else:
        msametf = None
        by_source = False

    use_file = rate_file

    if rename_f070:
        with pyfits.open(rate_file) as im:
            filt = im[0].header['FILTER']
            if filt == 'F070LP':
                im[0].header['FILTER'] = 'F100LP'

                new_file = new_filename(rate_file, c='b')
                msg = "Rename blocking filter F070LP to F100LP\n"
                msg += f"Rename {rate_file} > {new_file}"

                im.writeto(new_file, overwrite=True)

                os.remove(rate_file)

                use_file = new_file

                utils.log_comment(utils.LOGFILE, msg, verbose=True)

    use_prefix = use_file.split('_rate')[0]

    files = [use_file]
    files.sort()

    utils.log_comment(utils.LOGFILE, 'Reset DQ=4 flags', verbose=True)

    for _file in files:
        with pyfits.open(_file, mode='update') as im:
            # print(f'_file unset DQ=4')
            im['DQ'].data -= im['DQ'].data & 4
            im.flush()

    # Split into groups of 3 exposures
    groups = pipeline.exposure_groups(files=files, split_groups=True)

    print('Files:')
    print('======')
    print(yaml.dump(dict(groups)))

    # Single exposure groups
    single_exposure_groups = {}

    for g in groups:
        for exp, k in zip(groups[g], 'abcdefghijklmnopqrstuvwxyz'[:len(groups[g])]):
            gr = g.replace('-f', f'{k}-f').replace('-clear', f'{k}-clear')
            single_exposure_groups[gr] = [exp]

    print(yaml.dump(dict(single_exposure_groups)))

    pipes = []

    from jwst.assign_wcs.util import NoDataOnDetectorError

    source_ids = None

    pad = 0

    positive = False
    sources = None

    # Should just be one group....
    for g in groups:
        for exp, k in zip(groups[g], 'abcdefghijklmnopqrstuvwxyz'[:len(groups[g])]):
            mode = g.replace('-f', f'{k}-f').replace('-clear', f'{k}-clear')
            xmode = f'{mode}-fixed' if as_fixed else mode

            if sources is not None:
                source_ids = sources[g] #[3:6]
                if len(source_ids) < 1:
                    source_ids = None
            else:
                source_ids = None

            if os.path.exists(f'{xmode}.start'):
                print(f'Already started: {mode}')
                continue

            if outroot in ['macs0417-v1']:
                source_ids = None
                positive = False

            source_ids = None
            positive = False

            if not os.path.exists(f'{xmode}.slits.yaml'):#
                with open(f'{xmode}.start','w') as fp:
                    fp.write(time.ctime())

                if 0:
                    source_ids = sources[mode]

                if as_fixed:
                    for _file in single_exposure_groups[mode]:
                        with pyfits.open(_file, mode='update') as _im:
                            ORIG_EXPTYPE = _im[0].header['EXP_TYPE']
                            if ORIG_EXPTYPE != 'NRS_FIXEDSLIT':
                                print(f'Set {_file} MSA > FIXEDSLIT keywords')
                                _im[0].header['EXP_TYPE'] = 'NRS_FIXEDSLIT'
                                _im[0].header['APERNAME'] = 'NRS_S200A2_SLIT'
                                _im[0].header['OPMODE'] = 'FIXEDSLIT'
                                _im[0].header['FXD_SLIT'] = 'S200A2'
                                _im.flush()

                if extend_wavelengths:

                    if by_source & (msametf is not None):
                        # Run by individual source IDs
                        rate_file = single_exposure_groups[mode][0]

                        msa = msaexp.msa.MSAMetafile(msametf)
                        msa.merge_source_ids()
                        msa.write(prefix='', overwrite=True)

                        source_ids = msaexp.msa.get_msa_source_ids(rate_file)

                        with pyfits.open(rate_file, mode='update') as im:
                            if 'src' not in im[0].header['MSAMETFL']:
                                im[0].header['MSAMETFL'] = 'src_' + msametf

                            im.flush()

                        # Run by source_id
                        for source_id in source_ids:
                            done_files = glob.glob(f"*_{source_id}.fits")
                            if len(done_files) > 0:
                                print(f'Skip completed {done_files[0]}')

                            msametfl = msaexp.msa.pad_msa_metafile(
                                msametf,
                                pad=0,
                                positive_ids=True,
                                source_ids=[source_id],
                                slitlet_ids=None,
                                primary_sources=True,
                            )

                            try:
                                pipe = pipeline_extended.run_pipeline(
                                    rate_file,
                                    slit_index=0,
                                    all_slits=True,
                                    write_output=True,
                                    set_log=True,
                                    skip_existing_log=False,
                                    undo_flat=undo_flat,
                                )
                            except ValueError:
                                msg = f'Failed to process source_id={source_id}'
                                utils.log_comment(utils.LOGFILE, msg, verbose=True)
                                continue

                        # Set it back
                        with pyfits.open(rate_file, mode='update') as im:
                            if 'src' not in im[0].header['MSAMETFL']:
                                im[0].header['MSAMETFL'] = msametf

                            im.flush()

                    else:
                        pipe = pipeline_extended.run_pipeline(
                            single_exposure_groups[mode][0],
                            slit_index=0,
                            all_slits=True,
                            write_output=True,
                            set_log=True,
                            skip_existing_log=False,
                            undo_flat=undo_flat,
                        )

                        if as_fixed:
                            photom_file = f"{use_prefix}_fs-photom.fits"
                        else:
                            photom_file = f"{use_prefix}_photom.fits"
                        
                        print(f'Write {photom_file}')
                        pipe.write(photom_file)

                else:
                    try:
                        pipe = pipeline.NirspecPipeline(mode=xmode,
                                                    files=single_exposure_groups[mode],
                                                    source_ids=source_ids,
                                                    pad=pad,
                                                    positive_ids=positive # Ignore background slits
                                                   )

                        pipe.full_pipeline(run_extractions=False,
                                           initialize_bkg=False,
                                           load_saved=None,
                                           scale_rnoise=False,
                                           fix_rows=False,
                                           )

                    except NoDataOnDetectorError:
                        print('NoDataOnDetectorError - skip')
                        pipe = None
                    # continue

                if as_fixed:
                    for _file in single_exposure_groups[mode]:
                        with pyfits.open(_file, mode='update') as _im:
                            if ORIG_EXPTYPE == 'NRS_MSASPEC':
                                print(f'Reset {_file} FIXEDSLIT > MSA keywords')
                                _im[0].header['EXP_TYPE'] = 'NRS_MSASPEC'
                                _im[0].header['APERNAME'] = 'NRS_FULL_MSA'
                                _im[0].header['OPMODE'] = 'MSASPEC'
                                _im[0].header.pop('FXD_SLIT')
                                _im.flush()

                # pipes.append(pipe)
                del(pipe)

                os.remove(f'{xmode}.start')
                print(f'{xmode} - Done! {time.ctime()}')

            else:
                print(f'Already completed: {mode}')

            os.system(f'cat {mode}.log.txt >> {_NEW_LOGFILE}')

            # break

    utils.LOGFILE = _NEW_LOGFILE

    # Sync slitlets to S3
    if outroot.split('-')[0] in ['macs0417','macs1423','macs0416','abell370']:
        s3path = 'grizli-canucs/nirspec'
    else:
        s3path = 'msaexp-nirspec/extractions'

    if (outroot not in ['uncover-deep-v1']) & (1):
        msg = f'Sync slitlets to s3://{s3path}/slitlets/{outroot}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)

        os.system(f'aws s3 sync ./ s3://{s3path}/slitlets/{outroot}/ --exclude "*" --include "*phot.*" --include "*raw.*" --include "*photom.*" --acl public-read --quiet')

    if use_prefix != file_prefix:
        _USE_LOGFILE = os.path.join(WORKPATH, use_prefix + '_rate.log.txt')
        os.system(f"cp {_NEW_LOGFILE} {_USE_LOGFILE}")

    if os.path.exists(NIRSPEC_HOME):
        local_path = os.path.join(NIRSPEC_HOME, outroot)
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        msg = f'cp {WORKPATH}/{use_prefix}* {local_path}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        os.system(msg)

        msg = f'sudo chown -R ec2-user {local_path}/'
        utils.log_comment(utils.LOGFILE, msg, verbose=True)
        os.system(msg)

        if msametf is not None:
            msg = f'cp {WORKPATH}/{msametf} {local_path}/'
            utils.log_comment(utils.LOGFILE, msg, verbose=True)
            os.system(msg)

    utils.LOGFILE = _ORIG_LOGFILE

    if clean:
        print('Clean up')
        files = glob.glob('*')
        for file in files:
            print(f'rm {file}')
            os.remove(file)

        os.chdir(HOME)
        os.rmdir(WORKPATH)

    return 2

if __name__ == "__main__":
    run_one()


