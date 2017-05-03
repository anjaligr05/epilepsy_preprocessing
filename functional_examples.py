# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 01:31:51 2017

@author: chenym
"""

import func_preproc
from func_preproc import motion_correction as mc
from func_preproc import skullstrip4d as remove_skull
from func_preproc import smoothing_scaling as normalize
from func_preproc import masking
# an C-PAC example, which uses default values. files will be stored in func/ folder
func_preproc.NAME_CONV = 'accumulate'
masking(normalize(remove_skull(mc(mc('rest')))))

func_preproc.NAME_CONV = 'replace'
mc('rest','rest_mc','func_2')
remove_skull('rest_mc','rest_ss','func_2')
normalize('rest_ss', func_dir='func_2', smooth=6)
masking('rest_gms', func_dir='func_2')

from nilearn.plotting import plot_epi
from nilearn.image import mean_img
plot_epi(mean_img('func_2/rest_gms.nii.gz'), cut_coords=(-22,15,23))
plot_epi(mean_img('func_1/rest_mc_mc_ss_gms.nii.gz'), cut_coords=(-22,15,23))