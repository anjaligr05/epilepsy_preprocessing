# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:46 2017

@author: chenym
"""

# Python built-ins
import warnings
from shutil import copyfile

# customs
from utils import build_input_path, build_output_path, BrainImageFileNotFoundError

# I/O, nilearn, pypreprocess and nipy all depend on nibabel
import nibabel as nib

# sklearn package
try:
    from sklearn.externals.joblib import Memory # for caching!
    USE_CACHE = True
except ImportError:
    warnings.warn("error importing sklearn.externals.joblib.Memory. caching functions not available.")
    USE_CACHE = False

# nipy packages
try:
    from nipy import load_image, save_image
    from nipy.algorithms.registration import SpaceTimeRealign
    from nipy.algorithms.registration.groupwise_registration import SpaceRealign
except ImportError:
    warnings.warn("nipy modules are not available. 'nipy_spacetimerealign' and 'nipy_spacerealign' options of motion_correction can't be used. ")

# pypreprocess.realign package
try:
    from pypreprocess.realign import MRIMotionCorrection
except ImportError:
    warnings.warn("pypreprocess.realign module not available. 'pypreprocess_realign' option of motion_correction can't be used. ")

# nilearn modules
try:
    from nilearn.image import mean_img, smooth_img, clean_img, math_img
    from nilearn.masking import compute_epi_mask
    from nilearn.plotting import plot_roi
except ImportError:
    warnings.warn("can't import nilearn. most functions are dependent on these packages, except motion_correction with options 'nipy_spacetimerealign' and 'nipy_spacerealign'")

######################
# predefined runners #
######################

def run_default_cpac_functional_pipeline(rest):
    try:
        build_input_path(rest, 'func')
    except BrainImageFileNotFoundError:
        raise BrainImageFileNotFoundError("can't find input file func/%s.gii.gz. this function by default loads images from func/ folder, so make sure the folder is created and the functional image is in there. "%rest)
    
    return masking(smoothing_scaling(skullstrip4d(motion_correction(motion_correction(rest)))))
    
def run_configured_functional_pipeline(rest, func_dir, options):
    pass

#######################################
# functions for each individual steps #
#######################################

def skullstrip4d(name_in, func_dir = 'func/',
                 name_out = '', mask_name_out = '',
                 postfix = 'ss', mask_postfix = 'mask', name_conv = 'replace',
                 compute_epi_mask_kwargs = {}, **kwargs):
    """
    Create a 3D mask using nilearn masking utilities; apply the mask to motion
    corrected 4D functional images using numpy. Can pass extra parameters to
    nilearn.masking.compute_epi_mask by customizing compute_epi_mask_kwargs. 
    
    inputs:
        name_in: str. name of nii.gz file without extension. 
        name_out: the name of the output. can be unspecified (blank).
        mask_name_out: name for the mask file.
        postfix: add after name_in to build name_out, if name_out is unspecified. 
        mask_postfix: for unspecified mask_name_out. 
        func_dir: directory for functional images. 
        compute_epi_mask_kwargs: extra keyword arguments passed to 
                                 nilearn.masking.compute_epi_mask. 
        
    Return: the name of the skull stripped image
    """
    
    in_file = build_input_path(name_in, func_dir)
    name_out, out_file = build_output_path(name_in, name_out, func_dir, postfix, name_conv)
    name_out_mask, mask_out_file = build_output_path(name_in, mask_name_out, func_dir, mask_postfix, name_conv)    
    
    func = nib.load(in_file)
    
    print 'start skull stripping... input: %s' % in_file    

    mask = compute_epi_mask(func, **compute_epi_mask_kwargs)
    
    print "skull strip complete. check the plot to see if it's good! (if not, adjust parameters using compute_epi_mask_kwargs)"    
    
    # display mask
    plot_roi(mask, mean_img(func),title='mask for %s'%name_in)
    
    func = math_img('a[...,np.newaxis]*b', a = mask, b = func) # numpy broadcast
    
    nib.save(mask, mask_out_file)
    nib.save(func, out_file) 
    print 'mask and the skull stripped images are save to %s and %s, respectively. ' % (out_file, mask_out_file)
    
    return name_out

def masking(name_in, func_dir = 'func/', name_out = '', postfix = 'pp_mask', 
            name_conv = 'replace', **kwargs):
    """
    Create a binary mask for the 4D functional scan, based on the min values
    along t-axis. 
    
    Inputs:
        name_in: str. name of nii.gz file without extension. 
        name_out: name of the mask. can be unspecified. 
        postfix: add after name_in to build name_out, if name_out is unspecified.  
        func_dir: path to the directory that functional images will be stored. 
        
    Return: name of the mask
    """
    in_file = build_input_path(name_in, func_dir)
    name_out, out_file = build_output_path(name_in, name_out, func_dir, postfix, name_conv)
    
    func = nib.load(in_file)
    
    print "start creating a mask for input file: %s" % in_file    
    mask = math_img('(np.min(img,axis=-1) != 0).astype(int)', img=func)
    
    nib.save(mask, out_file)    
    print "mask saved: %s" % out_file
    
    return name_out

def smoothing_scaling(name_in, func_dir = 'func/',
                      name_out = '', postfix = 'gms', 
                      smooth = None, normalize = 10000,
                      name_conv = 'replace', **kwargs):
    """
    Has the ability to smooth and normalize the data. 
    
    For reference (and for what's it worth), after skull stripping, C-PAC
    normalizes image intensity values then goes straight ahead to create a mask; 
    meanwhile fcon_1000's pipeline does spatial smoothing (with FWHM=6), 
    grand mean scaling (which is the same as normalization in C-PAC), temporal
    filtering and detrending before masking. While smoothing and grand mean 
    scaling are achievable, FSL's filtering and detrending functions are
    different from the available nilearn functions. @See filtering

    By default we choose the C-PAC setting, 
        { 'smooth' : None, 'normalize' : 10000 }
    since it's newer and doesn't require smoothing or filtering. a fcon_1000's
    setting will look like this:
        { 'smooth' : 6, 'normalize' : 10000 }
    but to really reproduce their pipeline one have to deal with filters later. 
    
    Inputs:
        name_in: str. name of nii.gz file without extension. 
        name_out: name of the grand-mean-scaled functional scan
        postfix: add after name_in to build name_out, if name_out is unspecified.
        func_dir: the directory dedicated to functional files. 
        smooth: smooth parameter. Can be scalar, numpy.ndarray, 'fast' or None
        normalize: a number to multiply after dividing the 4D global mean. 
    
    Return: name of the smoothed, grandmean scaled image
    """
    
    in_file = build_input_path(name_in, func_dir)
    name_out, out_file = build_output_path(name_in, name_out, func_dir, postfix, name_conv)
    
    func = nib.load(in_file)

    # smooth
    if smooth:
        print "start smoothing the image, input: %s" % in_file
        
        mask = math_img('img!=0', img = func) # create a mask to preserve shape. 
        # why? because smoothing will generate a fuzzy outline around the 
        # brain image i.e., some of the zeros near the brain surface will be
        # non-zeros after gaussian filtering; so we need to remove them by
        # applying this mask. 
        # this mask should be the same as the one generated in skullstrip4d,
        # so alternatively one can load that file...       
        
        func = smooth_img(func, smooth)
        func = math_img('img2.astype(int)*img1', img1 = func, img2 = mask)

    # normalize
    if normalize:
        print "start normalizing the image, input: %s" % in_file
        # normalization here means to bring the global (4D) mean of every scan
        # (of different sessions or subjects) to a constant value. Specifically, 
        # FSL defines the global mean as the mean of all non-zero values. 
        # after all that's where it actually matters...

        func = math_img('img*(%f)/np.mean(img[img!=0])'%normalize, img=func)
#        func = math_img('img1*(%f)/np.mean(img1*img2)'%normalize, img1=func,img2=mask) 

    nib.save(func, out_file)
    print "grand mean scaled (and/or smoothed) image is saved: %s" % out_file
    return name_out

def filtering(name_in, func_dir = 'func/', name_out = '', postfix = 'pp',
              detrend = False, high_pass = None, low_pass = None,
              clean_kwargs = {'standardize':False, 't_r':2}, name_conv = 'replace', **kwargs):
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
        name_in, name_out, func_dir: str. file names and directories
        detrend: bool. whether to detrend the data
        high_pass: float or None. in Hz
        low_pass: float or None. in Hz
        clean_kwargs: extra keywords to nilean.image.clean_img
        
    Return: name of the filtered image
    """
    
    in_file = build_input_path(name_in, func_dir)
    name_out, out_file = build_output_path(name_in, name_out, func_dir, postfix, name_conv)
    
    func = nib.load(in_file)
    # band-pass filter, detrend
    if high_pass or low_pass or detrend:
        warnings.warn('This function may not do what you think it does. Read the description and view the output data before using them. ')
        dt = bool(detrend)
        lp = low_pass or None # if not low_pass then None
        hp = high_pass or None
        print "start cleaning. parameters: \n\tdetrend: %s\n\t lp: %s\thp: %s" % dt,lp,hp
        func = clean_img(func, detrend=dt, low_pass=lp, high_pass=hp, **clean_kwargs)

    nib.save(func, out_file)
    return name_out

def motion_correction(name_in, func_dir = 'func/', name_out = '', postfix = 'mc',
                      mc_alg = 'pypreprocess_realign', force_mean_reference = False,
                      mc_kwargs = {}, name_conv = 'replace', **kwargs):
    """
    Motion correction function. Offers 3 motion correction algorithms from 
    2 packages: SpaceRealign and SpaceTimeRealign from NiPy, realign from
    pypreprocess. All three algorithms are purely Python. 
    
    inputs:
        name_in: name of the resting state fMRI image. No extension. 
        name_out: name of the motion corrected resting state fMRI image. No extension.
        postfix: if name_out is False, use postfix to build name_out. 
        func_dir: the path to the directory where <name_in>.nii.gz can be found. 
                  Also the directory <rest_mc>.nii.gz will be saved. 
        mc_alg: keyword for algorithm choice. Options: 
                ['nipy_spacerealign', 
                 'nipy_spacetimerealign', 
                 'pypreprocess_realign']
        force_mean_reference: whether the motion correction should be aligned
                              with the first volume or the mean of all volumes. 
        mc_kwargs: extra kwargs to pass to motion correction algorithms
        
    return: 
        name of the motion corrected fMRI scan. 
    """

    nipyalgs = ['nipy_spacerealign', 'nipy_spacetimerealign']
    pypreprocessalgs = ['pypreprocess_realign']
    in_file = build_input_path(name_in, func_dir)
    name_out, out_file = build_output_path(name_in, name_out, func_dir, postfix, name_conv)
    
    print 'start motion correction process'
    if mc_alg in nipyalgs:
        print "using nipy's algorithm"
        motion_correction_nipy(in_file, out_file, mc_alg, mc_kwargs)
    elif mc_alg in pypreprocessalgs:
        print "using pypreprocess' algorithm"
        motion_correction_pypreprocess(in_file, out_file, name_out, func_dir, force_mean_reference, mc_kwargs)
    else:
        raise ValueError('option %s is not recognizable. '%mc_alg)

    return name_out

def motion_correction_pypreprocess(in_file, out_file, rest_mc, func_dir, force_mean_reference, mc_kwargs):
    """
    an attempt at motion correction using pypreprocess package. 
    
    inputs:
        in_file: path to the input file, which is a resting state fMRI image. 
        out_file: path to the future output file
        rest_mc: name of the future output file
        func_dir: directory of the future output file
        force_mean_reference: if evaluated True, adjust motion according to the 
                        mean image; otherwise adjust to the first volume. 
        mc_kwargs: extra parameters
    """
    # load using nibabel
    func = nib.load(in_file)
    
    if force_mean_reference: # calculate the mean and insert to the front
        func = math_img('np.insert(img, 0, np.mean(img, axis=-1), axis=3)', img = func)
#        func_mean = mean_img(func)
#        inserted_data = np.insert(func.get_data(), 0, func_mean.get_data(), axis=3)
#        func = nib.Nifti1Image(inserted_data, func.affine) # update func
        
    # instantiate realigner
    mrimc = MRIMotionCorrection(**mc_kwargs)

    # fit realigner
    if USE_CACHE:
        mem = Memory("func_preproc_cache")
        mrimc = mem.cache(mrimc.fit)(func)
    else:
        mrimc = mrimc.fit(func)
    
#    if force_mean_reference:
#        mrimc.vols_[0].pop(0)

    # write realigned files to disk
    result = mrimc.transform(concat=True)['realigned_images'][0]
    if force_mean_reference: # remove the first frame, which was the mean
        result = math_img('img[...,1:]', img = result)

    nib.save(result, out_file)
    

def motion_correction_nipy(in_file, out_file, mc_alg, mc_kwargs):
    """
    an attempt at motion correction using NiPy package. 
    
    inputs:
        in_file: Full path to the resting-state scan. 
        out_file: Full path to the (to be) output file. 
        mc_alg: can be either 'nipy_spacerealign' or 'nipy_spacetimerealign'
        mc_kwargs: extra parameters
    """
    
    alg_dict = {'nipy_spacerealign':(SpaceRealign, {}), 'nipy_spacetimerealign': 
        (SpaceTimeRealign, {'tr':2, 'slice_times':'asc_alt_2','slice_info':2})}
    # format: {'function_name':(function, kwargs), ...}

    # processing starts here    
    I = load_image(in_file)
    print 'source image loaded. '

    # initialize the registration algorithm
    reg = alg_dict[mc_alg][0](I, **alg_dict[mc_alg][1]) # SpaceRealign(I, 'tr'=2, ...)
    print 'motion correction algorithm established. '
    print 'estimating...'
    
    if USE_CACHE:
        mem = Memory("func_preproc_cache_2")
        mem.cache(reg.estimate)(refscan=None, **mc_kwargs)
    else:
        reg.estimate(refscan=None, **mc_kwargs)
#    reg.estimate(refscan=None)
    print 'estimation complete. Writing to file...'
    save_image(reg.resample(0), out_file)