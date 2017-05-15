# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 01:31:51 2017

@author: chenym

Below are a few examples on how to use func_preproc.py
"""

import func_preproc
from func_preproc import motion_correction
from func_preproc import skullstrip4d as remove_skull
from func_preproc import smoothing_scaling as normalize
from func_preproc import masking

"""
Below is an C-PAC example, which uses the default setting. This is the simplest
way of expressing serial commands/workflow, if the output of the first command
is the input of the second. Files will be stored in func/ folder. Make sure there
is already a "rest.nii.gz" file in that folder. 
"""
func_preproc.NAME_CONV = 'accumulate'
masking(normalize(remove_skull(motion_correction(motion_correction('rest')))))

"""
Below is an example imitating fcon_1000's pipeline. This time we uses more argument. 
"""
#func_preproc.NAME_CONV = 'replace'
#motion_correction('rest','rest_mc','func_2',mc_alg = 'nipy_spacerealign')
#remove_skull('rest_mc','rest_ss','func_2')
#normalize('rest_ss', func_dir='func_2', smooth=6)
#masking('rest_gms', func_dir='func_2')

from nilearn.plotting import plot_epi
from nilearn.image import mean_img
plot_epi(mean_img('func_2/rest_gms.nii.gz'), cut_coords=(-22,15,23))
plot_epi(mean_img('func_1/rest_mc_mc_ss_gms.nii.gz'), cut_coords=(-22,15,23))
plot_epi(mean_img('sample/rest_gms.nii.gz'), cut_coords=(-22,15,23))