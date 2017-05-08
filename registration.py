# -*- coding: utf-8 -*-
"""
Created on Fri May 05 03:09:54 2017

@author: chenym

some registration and co-registration
"""

import warnings
from utils import AllFeatures, build_image_path
import numpy as np
import nibabel as nib

REG_FUNC_OPTS = []

# nipy packages
try:
    from nipy import load_image, save_image
    from nipy.algorithms.registration import HistogramRegistration, resample, Affine
    REG_FUNC_OPTS.append('nipy')
except ImportError:
    warnings.warn("nipy modules are not available.")

# pypreprocess packages
try:
    from pypreprocess.coreg import Coregister
    from nilearn.image import resample_img
    from sklearn.externals.joblib import Memory # for caching!
    REG_FUNC_OPTS.append('pypreprocess')
except ImportError:
    warnings.warn("pypreprocess modules are not available.")

def build_reg_output_path(name_in, name_ref, reg_dir, name_conn):
    """
    build the paths to 3 output files for each registration process. 
    
    inputs:
        name_in, name_ref: name of files. str with no extention
        reg_dir: directory for load and save
        name_conn: string to bridge strings and show directions. e.g. 2, _to_
    return: a list of file paths, with orders like this:
            ['in2ref.nii.gz', 'in2ref.mat', 'ref2in.mat']
    """
    ls = []
    ls.append(build_image_path(name_in+name_conn+name_ref, reg_dir))
    ls.append(build_image_path(name_in+name_conn+name_ref, reg_dir, fileext='.mat'))
    ls.append(build_image_path(name_ref+name_conn+name_in, reg_dir, fileext='.mat'))
    
    return ls

def get_registration_function(registration_to_use):
    
    registration_options = {'nipy':affine_registration_nipy,
                            'pypreprocess':affine_registration_pypreprocess}

    if registration_to_use not in REG_FUNC_OPTS:
        err_msg = ("option %s is not valid. either it does not belong to"
        " the following list: %s, or there was an error during import. "
        ) % (registration_to_use, registration_options.keys())
        raise ValueError(err_msg)
    
    return registration_options[registration_to_use]

def concat_transforms(T1, T2, registration_to_use):
    """
    use this to avoid the conversion between affine matrices and
    (translation, rotation) vectors. 
    
    T1.dot(T2) for matrix (nipy) and T1 + T2 for vectors (pypreprocess)
    """   
    
    if registration_to_use == 'nipy':
        return T2.as_affine().dot(T1.as_affine())
    if registration_to_use == 'pypreprocess':
        print T1
        print T2
        return T1 + T2
    raise KeyError(registration_to_use)

def affine_registration_nipy(in_path, ref_path, out_path, 
                             in_ref_mat = '', ref_in_mat = '',
                             T = None, extra_params={}):
    """
    Affine registation and resampling. Use Histogram registration from nipy. 
    
    inputs:
        in_path: path to the source (input) image.
        ref_path: path to the target (reference) image.
        out_path: path to use to save the registered image. 
        in_ref_mat: if bool(in_ref_mat) is True, save the 4x4 transformation
                    matrix to a text file <in_ref_mat>. 
        ref_in_mat: if bool(ref_in_mat) is True, save the reverse of the 4x4
                    transformation matrix to a text file <ref_in_mat>. 
        T: affine transformation to use. if None, T will be estimated using 
           HistogramRegistration and optimizers; if type(T) is not Affine, 
           T = Affine(array=T)
        extra_params: extra parameters passing to HistogramRegistration,
                      HistogramRegistration.optimize, resample
        
    return T
    """

    source_image = load_image(in_path)
    target_image = load_image(ref_path)

    if T is None:
        print('assess the affine transformation using histogram registration. ')
        
#        R = HistogramRegistration(source_image, target_image)
        R = AllFeatures(HistogramRegistration,extra_params).run(source_image, target_image)
        
#        T = R.optimize('affine', optimizer='powell')
        T = AllFeatures(R.optimize,extra_params).run('affine', optimizer='powell')
        print('receive affine transformation %s' % T)

    else:
        if type(T) is not Affine:
            print('create Affine from T')
            T = Affine(array=T)
        print('using a predefined affine:\n%s\nwith a 4x4 matrix:\n%s\n' % (T, T.as_affine()))

#    It = resample(source_image, T.inv(), target_image)
    It = AllFeatures(resample,extra_params).run(source_image, T.inv(), target_image)

    # the second argument of resample takes an transformation from ref to mov
    # so that's why we need T.inv() here
    save_image(It, out_path)
    if in_ref_mat:
        np.savetxt(in_ref_mat, T.as_affine())
    if ref_in_mat:
        np.savetxt(ref_in_mat, T.inv().as_affine())
    
    return T

def affine_registration_pypreprocess(in_path, ref_path, out_path, 
                                     in_ref_mat = '', ref_in_mat = '',
                                     T = None, force_resample = False,
                                     extra_params={}):
    """
    Affine registation and resampling. Use Coregister from pypreprocess. 
    
    Coregister is designed for transformation between func and anat, so applying
    this function to mni standard space may not produce the best result.
    
    inputs:
        in_path: path to the source (input) image.
        ref_path: path to the target (reference) image.
        out_path: path to use to save the registered image. 
        in_ref_mat: if bool(in_ref_mat) is True, save the 4x4 transformation
                    matrix to a text file <in_ref_mat>. 
        ref_in_mat: if bool(ref_in_mat) is True, save the reverse of the 4x4
                    transformation matrix to a text file <ref_in_mat>. 
        T: specific transformation to use. if None, T will be estimated using 
           Coregister().fit; else numpy.array(T) will be used. T is an array
           of 6 elements; the first three represent translation, and the last
           three represent rotations. 
        force_resample: bool. whether or not to resample in an extra step. 
            by default pypreprocess does not resample data, which means we have
            to use nilearn's module to do that. also scaling is not one of the
            provided DoF/estimation parameters of pypreprocess, neither did I
            implement it myself. maybe check scipy.misc.imresize if scaling 
            needs to be implemented in the future. 
        extra_params: for Coregister()
        
    """
    source = nib.load(in_path)
    target = nib.load(ref_path)
    
#    coreg = Coregister()
    coreg = AllFeatures(Coregister, extra_params).run()
    
    if T is None:
        mem = Memory("affine_registration_pypreprocess_cache")
        coreg = mem.cache(coreg.fit)(target, source) # fit(target, source)
    else:
        T_ = np.array(T)
        if T_.size != 6 or T_.dtype != float:
            raise ValueError('T should either be None or ndarray with size 6 and dtype float')
        print('using predefined T = %s' % T)
        coreg.params_ = T_
    
    img = coreg.transform(source)[0]
    if force_resample: # no rescaling here
        img = mem.cache(resample_img)(img, target.affine, target.shape)
    nib.save(img, out_path)
    if in_ref_mat:
        np.savetxt(in_ref_mat,  coreg.params_)
    if ref_in_mat:
        np.savetxt(ref_in_mat, -coreg.params_)
    
    return coreg.params_