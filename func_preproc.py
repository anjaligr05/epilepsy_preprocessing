# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:46 2017

@author: chenym
"""

from nipy import load_image, save_image
from nipy.algorithms.registration import SpaceTimeRealign
from nipy.algorithms.registration.groupwise_registration import SpaceRealign
from utils import brain_image_path
from pypreprocess.realign import MRIMotionCorrection
from sklearn.externals.joblib import Memory # for caching!
import nilearn.image
import nibabel as nib
import numpy as np

def motion_correction(rest, rest_out, func_dir='func/', 
                      mc_alg='NiPy_SpaceRealign', ref_mean = True,
                      **kwargs):
    """
    Motion correction function. Offers 3 motion correction algorithms from 
    2 packages: SpaceRealign and SpaceTimeRealign from NiPy, realign from
    pypreprocessing. All three algorithms are purely Python. 
    
    args:
        rest: name of the resting state fMRI image. No extension. 
        rest_out: name of the motion corrected resting state fMRI image. No extension.
        func_dir: the path to the directory where <rest>.nii.gz can be found. 
                  Also the directory <rest_out>.nii.gz will be saved. 
        mc_alg: keyword for algorithm choice. Options: 
                ['NiPy_SpaceRealign', 
                 'NiPy_SpaceTimeRealign', 
                 'pypreprocessing_realign']
        ref_mean: whether the motion correction should be referred to the first
                  scan (t = 0), or the mean of all scans. 
    """
    nipyalgs = ['NiPy_SpaceRealign', 'NiPy_SpaceTimeRealign']
    pypreprocessalgs = ['pypreprocessing_realign']
    in_file = brain_image_path(func_dir, rest)
    out_file = brain_image_path(func_dir, rest_out, verify=False)
    
    if mc_alg in nipyalgs:
        motion_correction_nipy(in_file, out_file, mc_alg)
    elif mc_alg in pypreprocessalgs:
        motion_correction_pypreprocess(in_file, out_file, rest_out, func_dir, mc_alg, ref_mean)
    else:
        raise ValueError('option %s is not recognizable. '%mc_alg)

def motion_correction_pypreprocess(in_file, out_file, rest_out, func_dir, mc_alg, ref_mean):
    
    # load using nibabel
    func = nib.load(in_file)
    
    if ref_mean: # calculate the mean and insert to the front
        func_mean = nilearn.image.mean_img(func)
        func_mean_data = func_mean.get_data().astype(func.get_data_dtype())
        inserted_data = np.insert(func.get_data(), 0, func_mean_data, axis=3)
        func = nib.Nifti1Image(inserted_data, func.affine)
        
    # instantiate realigner
    mrimc = MRIMotionCorrection()

    # fit realigner
    mem = Memory("func_preproc_cache")
    mrimc = mem.cache(mrimc.fit)(func)
    
#    if ref_mean:
#        mrimc.vols_[0].pop(0)

    # write realigned files to disk
    result = mrimc.transform(func_dir, prefix = rest_out, ext='.nii.gz', concat=True)

    if ref_mean: # remove the first frame, which was the mean
        saved_img = nib.load(result['realigned_images'][0])
        final_img = nib.Nifti1Image(saved_img.get_data()[...,1:], saved_img.affine)
        nib.save(final_img, out_file)
    else: # need to rename the file
        pass
    
    

def motion_correction_nipy(in_file, out_file, mc_alg):
    """
    an attempt at functional prerprocessing. 
    
    args:.get
        in_file: name of the resting-state scan. Can be full path. 
        out_file: name of the (to be) output file. Can be full path. 
        mc_alg: can be either 'NiPy_SpaceRealign' or 'NiPy_SpaceTimeRealign'
    """
    
    alg_dict = {'NiPy_SpaceRealign':(SpaceRealign, {}), 'NiPy_SpaceTimeRealign': 
        (SpaceTimeRealign, {'tr':2, 'slice_times':'asc_alt_2','slice_info':2})}
    # format: {'function_name':(function, kwargs), ...}

    # processing starts here    
    I = load_image(in_file)
    print 'source image loaded. '

    # initialize the registration algorithm
    reg = alg_dict[mc_alg][0](I, **alg_dict[mc_alg][1]) # SpaceRealign(I, 'tr'=2, ...)
    print 'motion correction algorithm established. '
    print 'estimating...'
    
    reg.estimate(refscan=None)
    print 'estimation complete. Writing to file...'
    save_image(reg.resample(0), out_file)

def _delete_image_at_front():
    pass