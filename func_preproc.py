# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:58:46 2017

@author: chenym
"""

from nipy import load_image, save_image
from nipy.algorithms.registration import SpaceTimeRealign
from nipy.algorithms.registration.groupwise_registration import SpaceRealign
from utils import brain_image_path

def motion_correction(rest, rest_out, func_dir='func/', mc_alg='SpaceRealign'):
    """
    an attempt at functional prerprocessing. 
    
    args:
        rest: name of the resting-state scan. 
        rest_out:
        func_dir: working directory
        mc_alg: can be either 'SpaceRealign' or 'SpaceTimeRealign'
    """
    alg_dict = {'SpaceRealign':(SpaceRealign, {}), 
    'SpaceTimeRealign': (SpaceTimeRealign, 
                         {'tr':2, 'slice_times':'asc_alt_2','slice_info':2})}
    # format: {'function_name':(function, kwargs), ...}
    
    alg_tuple = alg_dict[mc_alg] # key error if wrong algorithm name
    
    # processing starts here    
    in_file = brain_image_path(func_dir, rest)
    out_file = brain_image_path(func_dir, rest_out, verify=False)
    I = load_image(in_file)
    print 'source image loaded. '

    # initialize the registration algorithm
    reg = alg_tuple[0](I, **alg_tuple[1])
    print 'motion correction algorithm established. '
    print 'estimating...'
    
    reg.estimate(refscan=None,borders=(4,4,4))
    print 'estimation complete. Writing to file...'
    save_image(reg.resample(0), out_file)