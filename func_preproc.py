# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:46 2017

@author: chenym
"""

from nipy import load_image, save_image
from nipy.algorithms.registration import SpaceTimeRealign
from nipy.algorithms.registration.groupwise_registration import SpaceRealign
from utils import build_image_path
from pypreprocess.realign import MRIMotionCorrection
from sklearn.externals.joblib import Memory # for caching!
from nilearn.image import mean_img, smooth_img, clean_img, math_img
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_roi
import nibabel as nib
import numpy as np
from shutil import copyfile
import warnings

def skullstrip4D(rest_in, rest_ss = '', ss_mask = '', func_dir = 'func/',
                 mask_kwargs = {}, **kwargs):
    """
    Create a 3D mask using nilearn masking utilities; apply the mask to motion
    corrected 4D functional image using numpy. One can pass extra parameters
    to nilearn.masking.compute_epi_mask through mask_kwargs. 
    
    inputs:
        rest_mc: the name of the motion corrected file. 
        rest_ss: the name of the skull stripped file. 
        func_dir: directory for functional images. 
        mask_kwargs: extra keyword arguments passed to nilearn.masking.compute_epi_mask
        
    Return: the name of the skull stripped image
    """
    in_file = build_image_path(func_dir, rest_in, check_exist = True)
    if not rest_ss:
        rest_ss = rest_in + '_ss'
    out_file = build_image_path(func_dir, rest_ss)
    
    if not ss_mask:
        ss_mask = rest_in + '_ss_mask'
    mask_out_file = build_image_path(func_dir, ss_mask)    
    
    func = nib.load(in_file)
    mask = compute_epi_mask(func, **mask_kwargs)
    
    # display mask
    plot_roi(mask, mean_img(func),title='mask for %s'%rest_in)
    
    func = math_img('a[...,np.newaxis]*b', a = mask, b = func) # numpy broadcast
    
    nib.save(mask, mask_out_file)
    nib.save(func, out_file)
    
    return rest_ss

def masking(rest_in, rest_pp_mask = '', func_dir = 'func/', **kwargs):
    """
    Create a binary mask for the 4D functional scan, based on the min values
    along t-axis. 
    
    Inputs:
        rest_in: name of the functional scan
        rest_pp_mask: name of the mask
        func_dir: path to the directory that functional images will be stored. 
        
    Return: name of the mask
    """
    in_file = build_image_path(func_dir, rest_in, check_exist = True)
    if not rest_pp_mask:
        rest_mask = rest_in + '_mask'
    out_file = build_image_path(func_dir, rest_pp_mask)
    
    func = nib.load(in_file)
    mask = math_img('(np.min(img,axis=-1) != 0).astype(int)', img=func)
    nib.save(mask, out_file)
    return rest_mask

def smoothing_scaling(rest_in, rest_gms = '', func_dir = 'func/', 
                      smooth = None, normalize = 10000, **kwargs):
    """
    Has the ability to smooth and normalize the data. 
    
    For reference (and for what's it worth), after skull stripping, C-PAC
    normalizes image intensity values then goes straught ahead to create mask; 
    meanwhile fcon_1000's pipeline does spatial smoothing (with FWHM=6), 
    grandmean scaling (which is the same as normalization in C-PAC), temporal
    filtering and detrending before masking. While smoothing and grandmean 
    scaling are achievable, FSL's filtering and detrending are different from 
    nilearn's. @See filtering In summary, for this part, to best reproduce
    C-PAC's pipeline, one should use parameters:
        { 'smooth' : None, 'normalize' : 10000 }

    while to replicate f_con one could use:
        { 'smooth' : 6, 'normalize' : 10000 }
    and deal with filtering later. By default we choose the C-PAC setting, 
    since it's newer and does less.
    
    Inputs:
        rest_in: name of the functional scan. 
        rest_gms: name of the preprocessed functional scan (i.e. after cleaning
                 the skull stripped scan <rest_ss>.nii.gz)
        func_dir: the directory dedicated to functional files. 
        smooth: smooth parameter. Can be scalar, numpy.ndarray, 'fast' or None
        normalize: a number to multiply after dividing the 4D global mean. 
    
    Return: name of the smoothed, grandmean scaled image
    """
    
    in_file = build_image_path(func_dir, rest_in, check_exist = True)
    if not rest_gms:
        rest_gms = rest_in + '_gms'
    out_file = build_image_path(func_dir, rest_gms)
    
    func = nib.load(in_file)
    
    # smooth
    if smooth:
        func = smooth_img(func, smooth) 

    # normalize
    if normalize:
        # normalization here means to bring the global (4D) mean of every scan
        # (of different sessions or subjects) to a constant value. Specifically, 
        # FSL defines the global mean as the mean of all non-zero values. 
        func = math_img('img*(%f)/np.mean(img[img!=0])'%normalize, img=func)        

    nib.save(func, out_file)
    return rest_gms

def filtering(rest_in, rest_filt = '', func_dir = 'func/', 
              detrend = False, high_pass = None, low_pass = None,
              clean_kwargs = {'standardize':False, 't_r':2}, **kwargs):
    """
    Filtering the data. Now, C-PAC has decided not to implement any of these
    features, and FSL always have weird definition of high-pass filtering by
    adding back DC values. Fcon_1000's pipeline took it futher by adding back
    DC value after detrending. Futhermore, nilearn has problem with their clean
    function, as seen here, https://github.com/nilearn/nilearn/issues/374#ref-issue-59767433,
    which affects the quality of filtering and detrending. In general, this 
    function probably won't work automatically and definitely won't behave like
    FSL or fcon. 
    
    Inputs: 
        rest_in, rest_filt, func_dir: str. file names and directories
        detrend: bool. whether to detrend the data
        high_pass: float or None 
        low_pass: float or None
        clean_kwargs: extra keywords to nilean.image.clean_img
        
    Return: name of the filtered image
    """
    
    in_file = build_image_path(func_dir, rest_in, check_exist = True)
    if not rest_filt:
        rest_filt = rest_in + '_pp'
    out_file = build_image_path(func_dir, rest_filt)
    
    func = nib.load(in_file)
    # band-pass filter, detrend
    if high_pass or low_pass or detrend:
        warnings.warn('This function may not do what you think it does. Read the description and view the output data before using them. ')
        dt = bool(detrend)
        lp = low_pass or None # if not low_pass then None
        hp = high_pass or None
        func = clean_img(func, detrend=dt, low_pass=lp, high_pass=hp, **clean_kwargs)

    nib.save(func, out_file)
    return rest_filt

def motion_correction(rest_in, rest_mc = '', func_dir = 'func/', 
                      mc_alg='NiPy_SpaceTimeRealign', mean_reference = True,
                      **kwargs):
    """
    Motion correction function. Offers 3 motion correction algorithms from 
    2 packages: SpaceRealign and SpaceTimeRealign from NiPy, realign from
    pypreprocess. All three algorithms are purely Python. 
    
    inputs:
        rest_in: name of the resting state fMRI image. No extension. 
        rest_mc: name of the motion corrected resting state fMRI image. No extension.
        func_dir: the path to the directory where <rest_in>.nii.gz can be found. 
                  Also the directory <rest_mc>.nii.gz will be saved. 
        mc_alg: keyword for algorithm choice. Options: 
                ['NiPy_SpaceRealign', 
                 'NiPy_SpaceTimeRealign', 
                 'pypreprocess_realign']
        mean_reference: whether the motion correction should be referred to 
                        the first volume (t = 0), or the mean of all volumes. 
        
    return: 
        name of the motion corrected fMRI scan. 
    """

    nipyalgs = ['NiPy_SpaceRealign', 'NiPy_SpaceTimeRealign']
    pypreprocessalgs = ['pypreprocess_realign']
    in_file = build_image_path(func_dir, rest_in, check_exist = True)
    if not rest_mc:
        rest_mc = rest_in + '_mc'
    out_file = build_image_path(func_dir, rest_mc)
    
    if mc_alg in nipyalgs:
        motion_correction_nipy(in_file, out_file, mc_alg)
    elif mc_alg in pypreprocessalgs:
        motion_correction_pypreprocess(in_file, out_file, rest_mc, func_dir, mean_reference)
    else:
        raise ValueError('option %s is not recognizable. '%mc_alg)

    return rest_mc

def motion_correction_pypreprocess(in_file, out_file, rest_mc, func_dir, mean_reference):
    """
    an attempt at motion correction using pypreprocess package. 
    
    inputs:
        in_file: path to the input file, which is a resting state fMRI image. 
        out_file: path to the future output file
        rest_mc: name of the future output file
        func_dir: directory of the future output file
        mean_reference: if evaluated True, adjust motion according to the 
                        mean image; otherwise adjust to the first volume. 
    """
    # load using nibabel
    func = nib.load(in_file)
    
    if mean_reference: # calculate the mean and insert to the front
        func_mean = mean_img(func)
        inserted_data = np.insert(func.get_data(), 0, func_mean.get_data(), axis=3)
        func = nib.Nifti1Image(inserted_data, func.affine) # update func
        
    # instantiate realigner
    mrimc = MRIMotionCorrection()

    # fit realigner
    mem = Memory("func_preproc_cache")
    mrimc = mem.cache(mrimc.fit)(func)
    
#    if mean_reference:
#        mrimc.vols_[0].pop(0)

    # write realigned files to disk
    result = mrimc.transform(func_dir, prefix = rest_mc, ext='.nii.gz', concat=True)
    saved_file = result['realigned_images'][0]
    if mean_reference: # remove the first frame, which was the mean
        saved_img = nib.load(saved_file)
        final_img = nib.Nifti1Image(saved_img.get_data()[...,1:], saved_img.affine)
        nib.save(final_img, out_file)
    else: # need to rename the file
        copyfile(saved_file, out_file)
    
    

def motion_correction_nipy(in_file, out_file, mc_alg):
    """
    an attempt at motion correction using NiPy package. 
    
    inputs:
        in_file: Full path to the resting-state scan. 
        out_file: Full path to the (to be) output file. 
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
    
    mem = Memory("func_preproc_cache_2")
    mem.cache(reg.estimate)(refscan=None)
#    reg.estimate(refscan=None)
    print 'estimation complete. Writing to file...'
    save_image(reg.resample(0), out_file)

if __name__ == '__main__':
#    masking(smoothing_scaling(skullstrip4D(motion_correction('rest'))))
    
#    motion_correction('rest','rest_mc_3',mc_alg='pypreprocess_realign',mean_reference=False)
    masking(smoothing_scaling(skullstrip4D('rest_mc_3')))
    from nilearn.plotting import plot_epi, plot_img
    plot_img('func/rest_mc_3_ss_mask.nii.gz',cut_coords=(-11,8,25))
    plot_epi('func/rest_mc_3_ss_gms.nii.gz',cut_coords=(-11,8,25))