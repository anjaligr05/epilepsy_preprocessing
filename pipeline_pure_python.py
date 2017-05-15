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
<func/rest_ss.nii.gz>. 


run_registration for registrating functional image to mni standard space. It 
first registers functional image to anatomical image, then registers anatomical
image to mni image. Finally, it registers the functional image to mni image by
concatinating the affine transformations obtained in the first two steps and
applying directly to the functional image. 

Usage:
run_registration('highres', 'example_func', 'standard', 'reg') will load
<reg/highres.nii.gz> as anatomical scan, <reg/examle_func.nii.gz> as example
functional scan, and <reg/standard.nii.gz> as mni template. 


Finally, the main function will run these two runners. It can be run in a
command line prompt too. 

Usage:
usage: pipeline_pure_python.py [-h] [-sl [SUBJECT_LIST]]
                               [-ac [ANAT_PREPROC_CONFIG]]
                               [-fc [FUNC_PREPROC_CONFIG]]
                               [-rc [REGISTRATION_CONFIG]]
                               main_dir
"""
import anat_preproc
import func_preproc
import registration
from registration import build_reg_output_path
import os
from utils import build_input_path, build_output_path, build_image_path
import argparse
import json
import shutil

def run_func_preproc(rest_name, func_dir, naming={}, func_to_run='default',
                     extra_params={}, **kwargs):
    """
    functional preprocessing workflow. will run functional preprocessing
    functions in series and save the intermediate outputs. 
    
    inputs:
        rest_name: the initial, original functional scan. name without extension. 
        func_dir: the working directory for functional preprocessing. 
        naming: the naming convention of the output files.
        func_to_run: a list of function names, or a single string "default".
            if it's "default", func_to_run = ['motion_correction', 
            'skullstrip4d', 'take_slice', 'smoothing_scaling', 'masking']
        extra_params: extra parameters to pass to functions. 
    """
    # all functions
    func_dict = {'motion_correction':func_preproc.motion_correction,
                 'skullstrip4d'     :func_preproc.skullstrip4d,
                 'take_slice'       :func_preproc.take_slice,
                 'smoothing_scaling':func_preproc.smoothing_scaling,
                 'masking'          :func_preproc.masking,
                 'filtering'        :func_preproc.filtering}
    # defaults
    if func_to_run == 'default':
        func_to_run = ['motion_correction', 'skullstrip4d', 'take_slice', 'smoothing_scaling', 'masking']
    if type(func_to_run) is not list:
        raise TypeError('func_to_run should be either "default" or a list (of strings). ')
    
    name_to_use = \
    {'naming_style'  : 'replace',
     'motion_correction_name_out':'', 'motion_correction_name_postfix':'mc',
     'skullstrip4d_name_out'     :'', 'skullstrip4d_name_postfix'     :'ss',
     'take_slice_name_out'       :'', 'take_slice_name_postfix'       :'example_func',
     'smoothing_scaling_name_out':'', 'smoothing_scaling_name_postfix':'gms',
     'filtering_name_out'        :'', 'filtering_name_postfix'        :'pp',
     'masking_name_out'          :'', 'masking_name_postfix'          :'pp_mask'
    }
    # update naming
    for key in naming.keys():
        name_to_use[key] = naming[key]
    
    name_in = rest_name
    naming_style = name_to_use['naming_style']
    input_list = []
    output_list = []
    return_dict = {}
    
    # starts
    print('running functional preprocessing. ')
    print("function to run: %s" % func_to_run)
    for func_name in func_to_run:
        # load output name
        name_out = name_to_use[func_name+'_name_out']
        name_postfix = name_to_use[func_name+'_name_postfix']
        # build path
        in_path = build_input_path(name_in, func_dir)
        name_out, out_path = build_output_path(name_in, name_out, func_dir, name_postfix, naming_style)
        
        input_list.append(name_in)        
        # run
        func_dict[func_name](in_path, out_path, extra_params=extra_params, **kwargs)
        output_list.append(name_out)
        return_dict[func_name] = out_path
        name_in = name_out
    
    print("functional preprocessing done. input output flow: ")
    print("function: %s\ninput: %s\noutput: %s"%(func_to_run,input_list,output_list)) 
    return return_dict

def run_registration(anat_name, func_name, mni_name, reg_dir, 
                     registration_to_use = 'nipy', _nb = '2',
                     extra_params = {}):
    """
    registration workflow. will register func to anat, then anat to stanard,
    and finally func to standard. 
    
    inputs:
        anat_name, func_name, mni_name: name of 3D scans, not extension. 
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
    anat_path = build_input_path(anat_name, reg_dir)
    func_path = build_input_path(func_name, reg_dir)
    mni_path = build_input_path(mni_name, reg_dir)
    
    # construct paths to future outputs
    op_1 = build_reg_output_path(func_name, anat_name, reg_dir, _nb)
    op_2 = build_reg_output_path(anat_name, mni_name, reg_dir, _nb)
    op_3 = build_reg_output_path(func_name, mni_name, reg_dir, _nb)
    
    # first register from functional space to anatomical (highres) space
    T1 = affine_registration(func_path, anat_path, op_1[0], op_1[1], op_1[2], 
                             extra_params=extra_params, **reg_params)
    
    # then register from anatomical (highres) space to mni (standard) space
    T2 = affine_registration(anat_path, mni_path, op_2[0], op_2[1], op_2[2], 
                             extra_params=extra_params, **reg_params)
                             
    # then register from functional space to mni (standard) space by directly
    # applying a transformation matrix
    T3 = registration.concat_transforms(T1, T2, registration_to_use)
    T3 = affine_registration(func_path, mni_path, op_3[0], op_3[1], op_3[2], 
                             T = T3, extra_params=extra_params, **reg_params)

def main(args):
    """
    main function. run functional, then create reg dir, copy files, and run
    registration. 
    
    args: containing input arguments
    """
    def _concate_main_dir(*otherstrings):
        return os.path.join(args.main_dir, *otherstrings)
    
    # read configs and override defaults
    anat_preproc_detail_config = {"anat_dir":"anat",
                                  "anat_skullstripped":"defaced_mprage.bse",
                                	"anat_postprocess": "defaced_mprage_brain"}
    func_preproc_detail_config = {'rest_name':'rest', 'func_dir':'func'}
    registration_detail_config = {'anat_name':'highres',
                                  'func_name':'example_func',
                                  'mni_name' :'standard',
                                  'template':'templates/standard.nii.gz',
                                  'reg_dir':'reg'}
    if args.func_preproc_config is not None:
        with open(args.func_preproc_config, 'r') as f:
            func_preproc_detail_config = json.load(f)
    if args.registration_config is not None:
        with open(args.registration_config, 'r') as f:
            registration_detail_config = json.load(f)
    if args.anat_preproc_config is not None:
        with open(args.anat_preproc_config, 'r') as f:
            anat_preproc_detail_config = json.load(f)
    
    # read subject list
    with open(_concate_main_dir(args.subject_list), 'r') as f:
        subject_list = [line.strip() for line in f if line.strip()]
    print 'receive subject list: %s' % subject_list
    for subject in subject_list:
        print 'running preprocessing pipeline for %s' % subject
        
        # post-anatomical preprocessing
        _kwanat = dict(anat_preproc_detail_config) # shallow copy
        _kwanat['anat_dir'] = _concate_main_dir(subject, _kwanat['anat_dir'])
        highres = anat_preproc.post_skullstrip(**_kwanat) # for registration
        
        # run functional preprocessing
        _kwfunc = dict(func_preproc_detail_config) # shallow copy
        _kwfunc['func_dir'] = _concate_main_dir(subject, _kwfunc['func_dir'])
        example_func_path = run_func_preproc(**_kwfunc)['take_slice']
        # run all func_preproc functions and store example_func for registration
        
        _kwreg = dict(registration_detail_config) # shallow copy
        _kwreg['reg_dir'] = _concate_main_dir(subject, _kwreg['reg_dir'])
        # create reg_dir; move files to reg_dir
        try:
            print "creating registration directory"
            os.mkdir(_kwreg['reg_dir'])
        except OSError:
            print "the directory may already be created. moving on..."
        # copy anat, func, mni files to new location
        old_path_dict = {'anat':highres,'mni':_kwreg['template'],'func':example_func_path}
        for key in old_path_dict:
            new_path = build_image_path(_kwreg[key+'_name'], _kwreg['reg_dir'])
            if new_path == old_path_dict[key]:
                print "same file, didn't copy. %s" % new_path
                continue
            shutil.copyfile(old_path_dict[key], new_path)
            print "copied %s file, from %s to %s" %(key, old_path_dict[key], new_path)
        _kwreg.pop('template') # don't need it anymore
        # run registration
        run_registration(**_kwreg)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("main_dir",
                        help='the main directory holding scans of all subjects')
    parser.add_argument('-sl', '--subject_list', nargs='?', 
                        default='subject_list.txt',
                        help='file listing all subjects')
    parser.add_argument('-ac','--anat_preproc_config', nargs='?', 
                        help="path to anatomical preprocessing's configurations")
    parser.add_argument('-fc','--func_preproc_config', nargs='?', 
                        help="path to functional preprocessing's configurations")
    parser.add_argument('-rc','--registration_config', nargs='?', 
                        help="path to registration's configurations")
    args = parser.parse_args()
    
    main(args)
        
        
