# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:33:16 2017

@author: chenym

This file contains two runners. run_func_preproc(...) for running (a selected
subset of) func_preproc's functions in series, that will read a configuration,
handle naming and I/Os, and adjust each function's parameters accordingly. 

Usage:
run_func_preproc('rest', 'func') will apply functional pre-processing on
image <func/rest.nii.gz> using default settings, and save intermediate outputs
to <func/> using default naming style. On the other hand, 
run_func_preproc(
    'rest', 'func',
    smooth = 6, force_mean_reference = True,
    func_to_run=['motion_correction','skullstrip4d','smoothing_scaling'],
    naming={'skullstrip4d_name_postfix':'ss_2'},
    extra_params={'compute_epi_mask':{'lower_cutoff':0.4, 'upper_cutoff':0.7}}
    )
will apply three steps: motion correction then skull stripping then smoothing;
the motion correction will be referenced to the mean, as indicated by
force_mean_reference = True; the skull stripping will include more "brain", as
indicated by compute_epi_mask's extra parameters, and the final skull stripped
image will be saved to <func/rest_ss_2.nii.gz> instead of the default
<func/rest_ss_.nii.gz>. 

run_registration for registrating functional image to mni standard space. It 
first registers functional image to anatomical image, then registers anatomical
image to mni image. Finally, it registers the functional image to mni image by
concatinating the affine transformation obtained in the first two steps and
applying directly to the functional image. 

Usage:
run_registration('highres', 'example_func', 'standard', 'reg') will load
<reg/highres.nii.gz> as anatomical scan, <reg/examle_func.nii.gz> as example
functional scan, and <reg/standard.nii.gz> as mni template. 

"""

import func_preproc
import registration
from registration import build_reg_output_path
import re
from utils import build_input_path, build_output_path

def run_func_preproc(rest, func_dir, naming={}, func_to_run='default',
                     extra_params={}, **kwargs):
    # all functions
    func_dict = {'motion_correction':func_preproc.motion_correction,
                 'skullstrip4d'     :func_preproc.skullstrip4d,
                 'smoothing_scaling':func_preproc.smoothing_scaling,
                 'masking'          :func_preproc.masking,
                 'filtering'        :func_preproc.filtering}
    # defaults
    if func_to_run == 'default':
        func_to_run = ['motion_correction', 'skullstrip4d', 'smoothing_scaling', 'masking']
    naming_to_use = \
    {'name_conv'  : 'replace',
     'extra_name' : {'skullstrip4d':['mask']}, 
     'motion_correction_name_out':'', 'motion_correction_name_postfix':'mc',
     'skullstrip4d_name_out'     :'', 'skullstrip4d_name_postfix'     :'ss',
     'skullstrip4d_mask_name_out':'', 'skullstrip4d_mask_name_postfix':'mask',
     'smoothing_scaling_name_out':'', 'smoothing_scaling_name_postfix':'gms',
     'filtering_name_out'        :'', 'filtering_name_postfix'        :'pp',
     'masking_name_out'          :'', 'masking_name_postfix'          :'pp_mask'
    }
    # update naming
    for key in naming.keys():
        naming_to_use[key] = naming[key]
    
    name_in = rest
    name_conv = naming_to_use['name_conv']
    input_list = []
    output_list = []
    
    # starts
    print("function to run: %s" % func_to_run)
    for func_name in func_to_run:
        # load output name
        name_out = naming_to_use[func_name+'_name_out']
        name_postfix = naming_to_use[func_name+'_name_postfix']
        # build path
        in_file = build_input_path(name_in, func_dir)
        name_out, out_file = build_output_path(name_in, name_out, func_dir, name_postfix, name_conv)
        # load extra path
        kwargs_copy = dict(kwargs)
        if func_name in naming_to_use['extra_name']:
            for extrakey in list(naming_to_use['extra_name'][func_name]):
                extrakey = re.sub('\A\_+|\_+\Z','',extrakey)
                _, extraout_file    = build_output_path(name_in,naming_to_use[
                   func_name+'_'+extrakey+'_name_out'],func_dir,naming_to_use[
                   func_name+'_'+extrakey+'_name_postfix'],name_conv)
                kwargs_copy[extrakey+'_'+'out_file'] = extraout_file
        input_list.append(name_in)
        func_dict[func_name](in_file, out_file, extra_params=extra_params, **kwargs_copy)
        output_list.append(name_out)        
        name_in = name_out
    
    print("functional preprocessing done. input output flow: ")
    print("function: %s\ninput: %s\noutput: %s"%(func_to_run,input_list,output_list))

def run_registration(anat, func, mni, reg_dir = 'reg', 
                 registration_to_use = 'nipy', _nb = '2', extra_params = {}):
    """
    registration workflow. will register func to anat, then anat to stanard,
    and finally func to standard. 
    
    inputs:
        anat, func, mni: name of the 3D scans, not extension. 
        reg_dir: directory
        registration_to_use: can choose from ['nipy', 'pypreprocess']
            will raise error if the package of interest can't be imported
            'pypreprocess' does not resample the source image into registered 
            space by default. in order to override that, one has to pass
            force_resample=True through extra_params, and the image still won't
            be scaled in the end. so use with caution. 
        _nb: str to combine source and target name. e.g. highres<2>standard
        extra_params: extra parameters to pass to functions. 
            format: extra_params = {'func_name':{'param_keyword':'value'}}
            e.g. extra_params = {'optimize':{'optimizer':'steepest'}} will 
            override the default optimizer for the optimize function. This works
            on classes too, e.g. extra_params =
            {'HistogramRegistration': {'smooth':6}} is equivalent to
            HistogramRegistration(smooth=6)
            positional arguments will be preserved and keyword arguments will be
            override. TypeError will raise if extra parameters did not fit.              
    """
    # choose which registration algorithm from which package to use    
    affine_registration = registration.get_registration_function(registration_to_use)
    # extra parameters for affine_registration_pypreprocess or affine_registration_nipy
    if affine_registration.__name__ in extra_params:
        reg_params = extra_params[affine_registration.__name__]
    else:
        reg_params = {}
    
    # construct paths to anat, func, mni file
    anat_path = build_input_path(anat, reg_dir)
    func_path = build_input_path(func, reg_dir)
    mni_path = build_input_path(mni, reg_dir)
    
    # construct paths to future outputs
    op_1 = build_reg_output_path(func, anat, reg_dir, _nb)
    op_2 = build_reg_output_path(anat, mni, reg_dir, _nb)
    op_3 = build_reg_output_path(func, mni, reg_dir, _nb)
    
    # first register from functional space to anatomical (highres) space
    T1 = affine_registration(func_path, anat_path, op_1[0], op_1[1], op_1[2], 
                             extra_params=extra_params, **reg_params)
    
    # then register from anatomical (highres) space to mni (standard) space
    T2 = affine_registration(anat_path, mni_path, op_2[0], op_2[1], op_2[2], 
                             extra_params=extra_params, **reg_params)
                             
    # then register from functional space to mni (standard) space by directly
    # applying a transformation matrix
    T3 = _concat_transforms(T1, T2, registration_to_use)
    T3 = affine_registration(func_path, mni_path, op_3[0], op_3[1], op_3[2], 
                             T = T3, extra_params=extra_params, **reg_params)

#def main(arg):
#	print "MENU\n"
#	print "1. Anatomical Preprocessing\n"
#	print "2. Functional Preprocessing\n"
#	print "3. Motion Correction\n"
#	choice = raw_input("Choose any of the above options: ")
#	inputfile = "Provide file here"
#	if choice=="1":
#		anatomical_preprocessing(inputfile)
#	elif choice=="2":
#		functional_preprocessing(inputfile)
#	else:
#		print "none"
#    if arg == '1': # anatomical preprocessing
#        post_skullstrip('defaced_mprage.bse', 'defaced_mprage_brain')
#    elif arg == '2': # functional preprocessing
#        motion_correction('rest', 'rest_mc_7', 
#                          func_dir='func',mc_alg='pypreprocessing_realign')
#        motion_correction('rest', 'rest_mc_5_', 
#                          func_dir='func',mc_alg='pypreprocessing_realign', ref_mean=False)
#        motion_correction('rest', 'rest_mc_4', 
#                           func_dir='func',mc_alg='NiPy_SpaceRealign')
if __name__=='__main__':
#	main(sys.argv[1])
#    run_registration('highres', 'example_func', 'standard', 'reg2')
    run_func_preproc('rest', 'func3')
