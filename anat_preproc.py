# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 16:28:01 2017

@author: chenym

These functions perform anatomical preprocessing, which is mostly skull 
stripping. Since skull stripping is done in Brainsuite, we don't have to do
much in Python. 
"""

import nibabel as nib
from nilearn.image import math_img

from shutil import copyfile
from utils import build_image_path

def post_skullstrip(anat_skullstripped, anat_postprocess, 
                    anat_mask = '', anat_postprocess_mask = 'mprage_surf',
                    anat_dir='anat/', flip_x=True): 
    """
    Here we assume the input file is skull stripped already, and therefore
    this function wouldn't do much except maybe flipping the orientation of 
    the scan. 
    
    args:
        anat_skullstripped: name of the anatomical scan file. This file is 
        assumed to be already skull stripped. 
        flip_x: whether to flip the x axis of the input scan. See: utils.flip_x_axis
        anat_out: the name of the output file, with extension. 
        anat_dir: current working directory
    """ 
    if anat_mask:
        _post_skullstrip(anat_mask, anat_postprocess_mask, anat_dir, flip_x)
    
    return _post_skullstrip(anat_skullstripped, anat_postprocess, anat_dir, flip_x)

def _post_skullstrip(anat_skullstripped, anat_postprocess, anat_dir, flip_x): 
    """
    Here we assume the input file is skull stripped already, and therefore
    this function wouldn't do much except maybe flipping the orientation of 
    the scan. 
    
    args:
        anat_skullstripped: name of the anatomical scan file. This file is 
        assumed to be already skull stripped. 
        flip_x: whether to flip the x axis of the input scan. See: utils.flip_x_axis
        anat_out: the name of the output file, with extension. 
        anat_dir: current working directory
    """ 
    in_file = build_image_path(anat_skullstripped, anat_dir, check_exist=True)
    out_file = build_image_path(anat_postprocess, anat_dir, check_exist=False)
    
    if not flip_x:
        if in_file == out_file:
            return out_file # nothing to do here
        # else if (not flip_x and anat != anat_out), copy the input file        
        copyfile(in_file, out_file)
        return out_file
    # else, we need to re-orientate
    I = math_img('np.flipud(data)', data = in_file)
    print 'anatomical data flipped'
    nib.save(I, out_file)
    return out_file