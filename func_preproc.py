# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:46 2017

@author: chenym

This file contains several functions that can be used to preprocess a functional
image. They are: 
motion_correction(...), motion_correction_nipy(...), and        <--see note 1.
motion_correction_pypreprocess(...) for motion correction, skullstrip4d(...)
for skull stripping, smoothing_scaling(...) for some spatial processing,
filtering(...) for temporal processing, and finally masking(...) for creating
a binary mask. The goal of this project is to reproduce fcon_1000 and C-PAC's
preprocessing pipeline without having their FSL and AFNI dependencies; thus the 
goal of these functions is to "echo" AFNI and FSL functions used in functional
preprocessing while utilizing open-source python packages such as nilearn, nipy
and pypreprocess. Below is a comparison between these functions and ANFI/FSL
equivalents. 

motion_correction --> 3dvolreg
skullstrip4d --> 3dAutomask -prefix $mask -dilate 1 $img
                 3dcalc -a $mask -b $img -expr 'a*b'
smoothing_scaling --> fslmaths -kernel gauss ${sigma} -fmean -mas $mask
                      fslmaths -ing 10000 -odt float ...
masking --> fslmaths -Tmin -bin ... -odt char                   <--see note 2.


It's possible to use functions in this file without installing all dependencies. 
An warning will appear if a package is not installed, but there won't be any
error in importing and using functions that does not depend on such package. 

Notes:
1. motion_correction(...) is a wrapper that will subsequently call either 
   motion_correction_nipy(...) or motion_correction_pypreprocess(...) depending
   on the parameter 'mc_alg'.
2. As noted in the docstring of filtering(...), this function has different
   behavior than fcon_1000's. 
"""

# Python built-ins
import warnings
import os

# customs
from utils import AllFeatures

# I/O, nilearn, pypreprocess and nipy all depend on nibabel
import nibabel as nib

# sklearn package
try:
    from sklearn.externals.joblib import Memory # for caching!
    USE_CACHE = True
except ImportError:
    warnings.warn("error importing sklearn.externals.joblib.Memory. caching functions not available.")
    USE_CACHE = False

# pypreprocess.realign package
try:
    from pypreprocess.realign import MRIMotionCorrection
except ImportError:
    warnings.warn("pypreprocess.realign module not available. 'pypreprocess_realign' option of motion_correction can't be used. ")

# nipy packages
try:
    from nipy import load_image, save_image
    from nipy.algorithms.registration import SpaceTimeRealign
    from nipy.algorithms.registration.groupwise_registration import SpaceRealign
    from nipy.io.nifti_ref import nifti2nipy, nipy2nifti
#    from nipy.core.api import Image
except ImportError:
    warnings.warn("nipy modules are not available. 'nipy_spacetimerealign' and 'nipy_spacerealign' options of motion_correction can't be used. ")

# nilearn modules
try:
    from nilearn.image import mean_img, smooth_img, clean_img, math_img, index_img
    from nilearn.masking import compute_epi_mask
    from nilearn.plotting import plot_roi
except ImportError:
    warnings.warn("can't import nilearn. most functions are dependent on these packages, except motion_correction with options 'nipy_spacetimerealign' and 'nipy_spacerealign'")


#######################################
# functions for each individual steps #
#######################################

def take_slice(in_file, out_path = '', slice_index = 7, **kwargs):
    """
    save a 3D image slice of a 4D functional image. for registration. 
    
    inputs:
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to output file
        slice_index: save image_data[...,slice_index], or t = slice_index
    
    return: a 3D image
    """
#    func = nib.load(in_file)
#    func = math_img('img[...,slice_index]', img=func)
    func = index_img(in_file, slice_index)
    if out_path:
        nib.save(func, out_path)
    return func

def skullstrip4d(in_file, out_path = '', extra_params={}, **kwargs):
    """
    Create a 3D mask using nilearn masking utilities; apply the mask to motion
    corrected 4D functional images using numpy. Can pass extra parameters to
    nilearn.masking.compute_epi_mask by customizing compute_epi_mask_kwargs. 
    A mask will be saved to name_out.mask.ext file for out_path = name_out.ext
    
    inputs:
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to where the skull stripped file will be saved
        extra_params: extra keyword arguments passed to nilearn.masking.compute_epi_mask. 
        
    return: the skull stripped image and its mask (in a tuple of two)
    """
    
    print 'computing mask...'    
    if 'compute_epi_mask' in extra_params:
        print "extra parameters are used for compute_epi_mask: %s" % extra_params['compute_epi_mask']
        mask = compute_epi_mask(in_file, **extra_params['compute_epi_mask'])
    else:
        mask = compute_epi_mask(in_file)
    
    stripped = math_img('a[...,np.newaxis]*b', a = mask, b = in_file) # numpy broadcast
    
    if not out_path:
        return stripped, mask
    
    print "skull strip complete. check the plot to see if it's good!"
    " (if not, adjust parameters using compute_epi_mask_kwargs)"  
    
    # display mask
    name_out,ext = os.path.splitext(out_path)
    if ext == '.gz':
        name_out,ext2 = os.path.splitext(name_out)
        ext = ext2+ext
    plot_roi(mask, mean_img(in_file),
             draw_cross=True,
             output_file=name_out+'.png',
             title='mask for %s'%out_path)
    print 'a visualization of the mask is saved at %s' % (name_out+'.png')
    
    nib.save(mask, name_out+'.mask'+ext)
    nib.save(stripped, out_path) 
    print 'mask and the skull stripped 4D scan are save to %s and %s, respectively. ' % (out_path, (name_out+'.mask'+ext))
    return stripped, mask

def masking(in_file, out_path = '', **kwargs):
    """
    Create a binary mask for the 4D functional scan, based on the min values
    along t-axis. 
    
    Inputs:
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to where the binary mask will be saved
        
    Return: out_path
    """
        
    mask = math_img('(np.min(img,axis=-1) != 0).astype(int)', img=in_file)
    if out_path:
        nib.save(mask, out_path)    
        print "mask saved: %s" % out_path
    return mask

def smoothing_scaling(in_file, out_path = '', smooth = None, normalize = 10000,
                      **kwargs):
    """
    Has the ability to smooth and normalize the data. 
    
    For reference (and for what's it worth), after skull stripping, C-PAC
    normalizes image intensity values then goes straight ahead to create a mask; 
    meanwhile fcon_1000's pipeline does spatial smoothing (with FWHM=6), 
    grand mean scaling (which is the same as normalization in C-PAC), temporal
    filtering and detrending before masking. While smoothing and grand mean 
    scaling are achievable, FSL's filtering and detrending functions are
    different from the available nilearn functions. @See filtering

    By default we choose the C-PAC setting: smooth=None, normalize=10000
    since it's newer and doesn't require smoothing or filtering. a fcon_1000's
    setting on the other hand will look like this: smooth=6, normalize=10000
    but to really reproduce their pipeline one have to deal with filters later. 
    
    Inputs:
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to where the spatially processed file will be saved
        smooth: nilearn.smooth_img's smooth parameter fwhm. Can be scalar,
                numpy.ndarray, 'fast' or None
        normalize: a number to multiply after dividing the 4D global mean. 
    
    Return: a smoothed and/or grandmean scaled image
    """

    # smooth
    if smooth:
        print 'apply smoothing with index %f' % smooth
        mask = math_img('img!=0', img = in_file) # create a mask to preserve 
        # shape. using gaussian kernel for smoothing will introduce a fuzzy 
        # outline around the edge and the shape and volume of the brain will be
        # altered, so we need to remove the extra edge by applying this mask. 
        # this mask should be the same as the one generated in skullstrip4d,
        # so alternatively one can load that file...       
        
        in_file = smooth_img(in_file, smooth)
        in_file = math_img('img2.astype(int)*img1', img1 = in_file, img2 = mask)

    # normalize
    if normalize:
        print "start normalizing the image to grand mean %f" % normalize
        # normalization here means to bring the global (4D) mean to a constant
        # value. only non-zeros values will be accounted when computing the mean. 
        # after all that's where it really matters...

        in_file = math_img('img*(%f)/np.mean(img[img!=0])'%normalize, img=in_file)
#        func = math_img('img1*(%f)/np.mean(img1*img2)'%normalize, img1=func,img2=mask) 

    if out_path:
        nib.save(in_file, out_path)
        print "grand mean scaled (and/or smoothed) image is saved: %s" % out_path
    return in_file

def filtering(in_file, out_path = '', detrend = False, 
              high_pass = None, low_pass = None, extra_params={}, **kwargs):
    """
    Automatically filter and detrend the data.
    
    Now, C-PAC has decided not to implement any of these features, and what
    fcon_1000 does is not traditional filtering (the DC values are always added
    back after high-pass filtering and detrending), making this function less
    crucial. Moreover, nilearn has its own issues, as seen here: https://github.com/nilearn/nilearn/issues/374#ref-issue-59767433,
    which means the result of this function can be fundamentally different from
    fcon_1000's, and the filter quality is inconsistent. As a result, one should
    always check the results before using them, or apply filters carefully and
    specially designed using other softwares. 
    
    Inputs: 
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to where the temporally processed file will be saved
        detrend: bool. whether to detrend the data
        high_pass: float or None. in Hz
        low_pass: float or None. in Hz
        extra_params: extra keywords to nilean.image.clean_img
        
    Return: the filtered image
    """

    # band-pass filter, detrend
    if high_pass or low_pass or detrend:
        warnings.warn('This function may not do what you think it does. Read the description and view the output data before using them. ')
        dt = bool(detrend)
        lp = low_pass or None # if not low_pass then None
        hp = high_pass or None
        print "start cleaning. parameters: \n\tdetrend: %s\n\t lp: %s\thp: %s" % dt,lp,hp
        print "Note: extra_params['clean_img'] may change these settings. "
        out_file = AllFeatures(clean_img, extra_params).run(
        in_file, detrend=dt, low_pass=lp, high_pass=hp, standardize=False, t_r=2)
#        func = clean_img(func, detrend=dt, low_pass=lp, high_pass=hp, standardize=False, t_r=2)
    if out_path:
        nib.save(out_file, out_path)
    return out_file

def motion_correction(in_file, out_path = '', mc_alg = 'pypreprocess_realign',
                      force_mean_reference = False, extra_params={}, **kwargs):
    """
    Motion correction function. Offers 3 motion correction algorithms from 
    2 packages: SpaceRealign and SpaceTimeRealign from NiPy, realign from
    pypreprocess. All three algorithms are purely Python. 
    
    inputs:
        in_file: path to the input file or input file loaded as an nibabel image. 
        out_path: path to where the motion corrected file will be saved
        mc_alg: keyword for algorithm choice. Options: 
                ['nipy_spacerealign', 
                 'nipy_spacetimerealign', 
                 'pypreprocess_realign']
        force_mean_reference: whether the motion correction should be aligned
                              with the first volume or the mean of all volumes. 
        extra_params: extra kwargs to pass to motion correction algorithms
        
    return: 
        the motion corrected fMRI scan. 
    """

    nipyalgs = ['nipy_spacerealign', 'nipy_spacetimerealign']
    pypreprocessalgs = ['pypreprocess_realign']
    
    print 'start motion correction process'
    if mc_alg in nipyalgs:
        print "using nipy's algorithm"
        return motion_correction_nipy(in_file, out_path, mc_alg, extra_params)
    elif mc_alg in pypreprocessalgs:
        print "using pypreprocess' algorithm"
        return motion_correction_pypreprocess(in_file, out_path,
                                              force_mean_reference, extra_params)
    
    raise ValueError('option %s is not recognizable. '%mc_alg)

def motion_correction_pypreprocess(in_file, out_path, force_mean_reference,
                                   extra_params={}):
    """
    an attempt at motion correction using pypreprocess package. 
    
    inputs:
        in_file: path to the input file or input file loaded as an nibabel image.  
        out_path: path to the future output file
        force_mean_reference: if evaluated True, adjust motion according to the 
                        mean image; otherwise adjust to the first volume. 
        extra_params: extra parameters to MRIMotionCorrection
    return: the motion corrected image
    """

    if force_mean_reference: # calculate the mean and insert to the front
        print('motion correction referenced to mean!')
        in_file = math_img('np.insert(img, 0, np.mean(img, axis=-1), axis=3)', img = in_file)
    else:
        print('motion correction referenced to the first slice.')
        
    # instantiate realigner
    if 'MRIMotionCorrection' in extra_params:
        print 'extra parameters are used for MRIMotionCorrection: %s' % extra_params['MRIMotionCorrection']
        mrimc = MRIMotionCorrection(**extra_params['MRIMotionCorrection'])
    else:
        mrimc = MRIMotionCorrection()

    # fit realigner
    if USE_CACHE:
        mem = Memory("func_preproc_cache")
        mrimc = mem.cache(mrimc.fit)(in_file)
    else:
        mrimc = mrimc.fit(in_file)

    # write realigned files to disk
    result = mrimc.transform(concat=True)['realigned_images'][0]
    if force_mean_reference: # remove the first frame, which was the mean
        result = math_img('img[...,1:]', img = result)

    if out_path:
        nib.save(result, out_path)
    return result
    

def motion_correction_nipy(in_file, out_path, mc_alg, extra_params={}):
    """
    an attempt at motion correction using NiPy package. 
    
    inputs:
        in_file: Full path to the resting-state scan. 
        out_path: Full path to the (to be) output file. 
        mc_alg: can be either 'nipy_spacerealign' or 'nipy_spacetimerealign'
        extra_params: extra parameters to SpaceRealign, SpaceTimeRealign, estimate
    return: the motion corrected image
    """
    
    alg_dict = {'nipy_spacerealign':(SpaceRealign, {}), 'nipy_spacetimerealign': 
        (SpaceTimeRealign, {'tr':2, 'slice_times':'asc_alt_2','slice_info':2})}
    # format: {'function_name':(function, kwargs), ...}

    # processing starts here
    if type(in_file) in nib.all_image_classes:
        I = nifti2nipy(in_file) # assume Nifti1Image
    else:
        I = load_image(in_file)
    print 'source image loaded. '

    # initialize the registration algorithm
    reg = AllFeatures(alg_dict[mc_alg][0], extra_params).run(I, **alg_dict[mc_alg][1])
#    reg = alg_dict[mc_alg][0](I, **alg_dict[mc_alg][1]) # SpaceTimeRealign(I, tr=2, ...)
    print 'motion correction algorithm established. '
    print 'estimating...'
    
    if USE_CACHE:
        mem = Memory("func_preproc_cache_2")
        mem.cache(AllFeatures(reg.estimate, extra_params).run)(refscan=None)
#        mem.cache(reg.estimate)(refscan=None)
    else:
        AllFeatures(reg.estimate, extra_params).run(refscan=None)
#        reg.estimate(refscan=None)

    print 'estimation complete. Writing to file...'
    result = reg.resample(0)
    if out_path:
        save_image(result, out_path)
    return nipy2nifti(result)