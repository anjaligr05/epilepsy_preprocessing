"""
Usage: 
python pipeline_anjali_work.py -f SUB_190011/func/rest.nii
      -t template_MNI152_2mm.nii.gz --TR 2 -s SUB_190011
      --subjects_dir fsdata --slice_times 0 17 1 18 2 19 3 20 4 21 5 22 6 23
      7 24 8 25 9 26 10 27 11 28 12 29 13 30 14 31 15 32 16 -o .
This workflow takes resting timeseries and a Nifti file corresponding
to it and preprocesses it to produce timeseries coordinates or grayordinates.
"""
from __future__ import division, unicode_literals
from builtins import open, range, str

import os

from nipype.interfaces.base import CommandLine
CommandLine.set_default_terminal_output("allatonce")
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.algorithms.misc import TSNR
from nipype.interfaces.utility import Rename, Merge, IdentityInterface
from nipype.utils.filemanip import filename_to_list
from nipype.interfaces.io import DataSink, FreeSurferSource

import numpy as np
import scipy as sp
import nibabel as nb


imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]
def anat_preproc(filename):

	from nilearn.datasets import fetch_localizer_button_task
	from nilearn.datasets import load_mni152_template
	from nilearn import plotting
	from nilearn.image import resample_to_img
	from nilearn.image import load_img

	template = load_mni152_template()
	localizer_dataset = fetch_localizer_button_task(get_anats=True)
	localizer_tmap_filename = localizer_dataset.tmaps[0]
	localizer_anat_filename = localizer_dataset.anats[0]
	
	resampled_localizer_tmap = resample_to_img(filename, template)
	
	tmap_img = load_img(filename)
	original_shape = tmap_img.shape
	original_affine = tmap_img.affine
	resampled_shape = resampled_localizer_tmap.shape
	resampled_affine = resampled_localizer_tmap.affine
	template_img = load_img(template)
	template_shape = template_img.shape
	template_affine = template_img.affine
	print("""Shape comparison:
	-Original t-map image shape: {0}
	-Resampled t-map image shape: {1}
	-Template image shape: {2}
	""".format(original_shape, resampled_shape, template_shape))
	print("""Affine comparison:
	-Original t-map image affine:\n{0}
	-Resampled t-map image:\n{0}
	-Template image affine:\n{2}
	""".format(original_affine,resampled_affine, template_affine))
	
	plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="t-map on original anat")
	plotting.plot_stat_map(resampled_localizer_tmap,
                       bg_img=template,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="Resampled t-map on MNI template anat")
	plotting.show()	

'''input agrs: file path to nifti 4D time series
output args: returns a 3d nifti file'''
def avg_median(filename_in):
	from nilearn.image import load_img
""" Create the main preprocessing workflow """

def create_workflow(files, target_file, subject_id, TR, slice_times, norm_threshold=1, num_components=5, vol_fwhm=None, surf_fwhm=None, lowpass_freq=-1, highpass_freq=-1, subjects_dir=None, sink_directory=os.getcwd(), target_subject=['fsaverage3','fsaverage4'],name='resting'):
	'''create workflow object, with arg name passed'''
	wf=Workflow(name=name)
	
	name_unique = MapNode(Rename(format_string='rest_%(run)02d'),
                          iterfield=['in_file', 'run'],
                          name='rename')
    	name_unique.inputs.keep_ext = True
   	name_unique.inputs.run = list(range(1, len(files) + 1))
    	name_unique.inputs.in_file = files

    	realign = Node(interface=spm.Realign(), name="realign")
    	realign.inputs.jobtype = 'estwrite'

    	num_slices = len(slice_times)
    	slice_timing = Node(interface=spm.SliceTiming(), name="slice_timing")
    	slice_timing.inputs.num_slices = num_slices
    	slice_timing.inputs.time_repetition = TR
    	slice_timing.inputs.time_acquisition = TR - TR / float(num_slices)
    	slice_timing.inputs.slice_order = (np.argsort(slice_times) + 1).tolist()
    	slice_timing.inputs.ref_slice = int(num_slices / 2)

    	# Computing TSNR on realigned data regressing polynomials upto order 2
   	tsnr = MapNode(TSNR(regress_poly=2), iterfield=['in_file'], name='tsnr')
    	wf.connect(slice_timing, 'timecorrected_files', tsnr, 'in_file')	

    	# Computing the median image across runs
    	calc_median = Node(Function(input_names=['in_files'],
                                output_names=['median_file'],
                                function=median,
                                imports=imports),
                       name='median')
    	wf.connect(tsnr, 'detrended_file', calc_median, 'in_files')

    	"""Segment and De register"""
    	registration = create_reg_workflow(name='registration')
    	wf.connect(calc_median, 'median_file', registration, 'inputspec.mean_image')
    	registration.inputs.inputspec.subject_id = subject_id
    	registration.inputs.inputspec.subjects_dir = subjects_dir
    	registration.inputs.inputspec.target_image = target_file

    	art = Node(interface=ArtifactDetect(), name="art")
    	art.inputs.use_differences = [True, True]
    	art.inputs.use_norm = True
    	art.inputs.norm_threshold = norm_threshold
    	art.inputs.zintensity_threshold = 9
    	art.inputs.mask_type = 'spm_global'
    	art.inputs.parameter_source = 'SPM'

    	"""Connect all the nodes together """
    	wf.connect([(name_unique, realign, [('out_file', 'in_files')]),
                (realign, slice_timing, [('realigned_files', 'in_files')]),
                (slice_timing, art, [('timecorrected_files', 'realigned_files')]),(realign, art, [('realignment_parameters', 'realignment_parameters')]),])

	def selectindex(files, idx):
        	import numpy as np
        	from nipype.utils.filemanip import filename_to_list, list_to_filename
        	return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

    	mask = Node(fsl.BET(), name='getmask')
    	mask.inputs.mask = True
    	wf.connect(calc_median, 'median_file', mask, 'in_file')
    	# get segmentation in normalized functional space

    	def merge_files(in1, in2):
        	out_files = filename_to_list(in1)
        	out_files.extend(filename_to_list(in2))
		return out_files 

	'''Compute motion regressos'''
	# Compute motion regressors
	motreg = Node(Function(input_names=['motion_params', 'order',
                                        'derivatives'],
                           output_names=['out_files'],
                           function=motion_regressors,
                           imports=imports),
                  name='getmotionregress')
	wf.connect(realign, 'realignment_parameters', motreg, 'motion_params')

	# Create a filter to remove motion and art confounds
    	createfilter1 = Node(Function(input_names=['motion_params', 'comp_norm',
                                               'outliers', 'detrend_poly'],
                                  output_names=['out_files'],
                                  function=build_filter1,
                                  imports=imports),
                         name='makemotionbasedfilter')
    	createfilter1.inputs.detrend_poly = 2
   	wf.connect(motreg, 'out_files', createfilter1, 'motion_params')
    	wf.connect(art, 'norm_files', createfilter1, 'comp_norm')
    	wf.connect(art, 'outlier_files', createfilter1, 'outliers')

    	filter1 = MapNode(fsl.GLM(out_f_name='F_mcart.nii',
                              out_pf_name='pF_mcart.nii',
                              demean=True),
                      iterfield=['in_file', 'design', 'out_res_name'],
                      name='filtermotion')

    	wf.connect(slice_timing, 'timecorrected_files', filter1, 'in_file')
    	wf.connect(slice_timing, ('timecorrected_files', rename, '_filtermotart'),filter1, 'out_res_name')
	wf.connect(createfilter1, 'out_files', filter1, 'design')

    	createfilter2 = MapNode(Function(input_names=['realigned_file', 'mask_file',
                                                  'num_components',
                                                  'extra_regressors'],
                                     output_names=['out_files'],
                                     function=extract_noise_components,
                                     imports=imports),
                            iterfield=['realigned_file', 'extra_regressors'],
                            name='makecompcorrfilter')
    	createfilter2.inputs.num_components = num_components

    	wf.connect(createfilter1, 'out_files', createfilter2, 'extra_regressors')
    	wf.connect(filter1, 'out_res', createfilter2, 'realigned_file')
    	wf.connect(registration, ('outputspec.segmentation_files', selectindex, [0, 2]),
               createfilter2, 'mask_file')

    	filter2 = MapNode(fsl.GLM(out_f_name='F.nii',
                              out_pf_name='pF.nii',
                              demean=True),
                      iterfield=['in_file', 'design', 'out_res_name'],
                      name='filter_noise_nosmooth')
    	wf.connect(filter1, 'out_res', filter2, 'in_file')
    	wf.connect(filter1, ('out_res', rename, '_cleaned'),
               filter2, 'out_res_name')
    	wf.connect(createfilter2, 'out_files', filter2, 'design')
    	wf.connect(mask, 'mask_file', filter2, 'mask')

    	bandpass = Node(Function(input_names=['files', 'lowpass_freq',
                                          'highpass_freq', 'fs'],
                             output_names=['out_files'],
                             function=bandpass_filter,
                             imports=imports),
                    name='bandpass_unsmooth')
    	bandpass.inputs.fs = 1. / TR
    	bandpass.inputs.highpass_freq = highpass_freq
    	bandpass.inputs.lowpass_freq = lowpass_freq
    	wf.connect(filter2, 'out_res', bandpass, 'files')

	'''Smooth the functional images'''
	smooth = Node(interface=spm.Smooth(), name="smooth")
	smooth.inputs.fwhm = vol_fwhm

    	wf.connect(bandpass, 'out_files', smooth, 'in_files')

    	collector = Node(Merge(2), name='collect_streams')
    	wf.connect(smooth, 'smoothed_files', collector, 'in1')
    	wf.connect(bandpass, 'out_files', collector, 'in2')

	"""Transform the remaining images, first to anatomical then to target"""
	warpall = MapNode(ants.ApplyTransforms(), iterfield=['input_image'],
                      name='warpall')
    	warpall.inputs.input_image_type = 3
    	warpall.inputs.interpolation = 'Linear'
    	warpall.inputs.invert_transform_flags = [False, False]
    	warpall.inputs.terminal_output = 'file'
    	warpall.inputs.reference_image = target_file
    	warpall.inputs.args = '--float'
	warpall.inputs.num_threads = 1

	# transform to target
    	wf.connect(collector, 'out', warpall, 'input_image')
    	wf.connect(registration, 'outputspec.transforms', warpall, 'transforms')

    	mask_target = Node(fsl.ImageMaths(op_string='-bin'), name='target_mask')

    	wf.connect(registration, 'outputspec.anat2target', mask_target, 'in_file')

    	maskts = MapNode(fsl.ApplyMask(), iterfield=['in_file'], name='ts_masker')
    	wf.connect(warpall, 'output_image', maskts, 'in_file')
	wf.connect(mask_target, 'out_file', maskts, 'mask_file')

	# map to surface
    	# extract aparc+aseg ROIs
   	# extract subcortical ROIs
    	# extract target space ROIs
    	# combine subcortical and cortical rois into a single cifti file

    	#######
    	# Convert aparc to subject functional space

    	# Sample the average time series in aparc ROIs
    	sampleaparc = MapNode(freesurfer.SegStats(default_color_table=True),
                          iterfield=['in_file', 'summary_file',
                                     'avgwf_txt_file'],
                          name='aparc_ts')
    	sampleaparc.inputs.segment_id = ([8] + list(range(10, 14)) + [17, 18, 26, 47] +
                                     list(range(49, 55)) + [58] + list(range(1001, 1036)) +
                                     list(range(2001, 2036)))

    	wf.connect(registration, 'outputspec.aparc',
               sampleaparc, 'segmentation_file')
    	wf.connect(collector, 'out', sampleaparc, 'in_file')

    	def get_names(files, suffix):
        	"""Generate appropriate names for output files
        	"""
        	from nipype.utils.filemanip import (split_filename, filename_to_list,
                                            list_to_filename)
        	out_names = []
        	for filename in files:
            		_, name, _ = split_filename(filename)
            		out_names.append(name + suffix)
        	return list_to_filename(out_names)

    	wf.connect(collector, ('out', get_names, '_avgwf.txt'),
               sampleaparc, 'avgwf_txt_file')
    	wf.connect(collector, ('out', get_names, '_summary.stats'),
               sampleaparc, 'summary_file')

    	# Sample the time series onto the surface of the target surface. Performs
    	# sampling into left and right hemisphere
    	target = Node(IdentityInterface(fields=['target_subject']), name='target')
    	target.iterables = ('target_subject', filename_to_list(target_subject))

    	samplerlh = MapNode(freesurfer.SampleToSurface(),
                        iterfield=['source_file'],
                        name='sampler_lh')
    	samplerlh.inputs.sampling_method = "average"
    	samplerlh.inputs.sampling_range = (0.1, 0.9, 0.1)
    	samplerlh.inputs.sampling_units = "frac"
    	samplerlh.inputs.interp_method = "trilinear"
    	samplerlh.inputs.smooth_surf = surf_fwhm
    	# samplerlh.inputs.cortex_mask = True
    	samplerlh.inputs.out_type = 'niigz'
    	samplerlh.inputs.subjects_dir = subjects_dir

    	samplerrh = samplerlh.clone('sampler_rh')

    	samplerlh.inputs.hemi = 'lh'
    	wf.connect(collector, 'out', samplerlh, 'source_file')
    	wf.connect(registration, 'outputspec.out_reg_file', samplerlh, 'reg_file')
    	wf.connect(target, 'target_subject', samplerlh, 'target_subject')

    	samplerrh.set_input('hemi', 'rh')
    	wf.connect(collector, 'out', samplerrh, 'source_file')
    	wf.connect(registration, 'outputspec.out_reg_file', samplerrh, 'reg_file')
    	wf.connect(target, 'target_subject', samplerrh, 'target_subject')

    	# Combine left and right hemisphere to text file
    	combiner = MapNode(Function(input_names=['left', 'right'],
                                output_names=['out_file'],
                                function=combine_hemi,
                                imports=imports),
                       iterfield=['left', 'right'],
                       name="combiner")
    	wf.connect(samplerlh, 'out_file', combiner, 'left')
    	wf.connect(samplerrh, 'out_file', combiner, 'right')

    	# Sample the time series file for each subcortical roi
    	ts2txt = MapNode(Function(input_names=['timeseries_file', 'label_file',
                                           'indices'],
                              output_names=['out_file'],
                              function=extract_subrois,
                              imports=imports),
                     iterfield=['timeseries_file'],
                     name='getsubcortts')
    	ts2txt.inputs.indices = [8] + list(range(10, 14)) + [17, 18, 26, 47] +\
        	list(range(49, 55)) + [58]
    	ts2txt.inputs.label_file = \
        	os.path.abspath(('OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_'
                         '2mm_v2.nii.gz'))
    	wf.connect(maskts, 'out_file', ts2txt, 'timeseries_file')

    	######

    	substitutions = [('_target_subject_', ''),
                     ('_filtermotart_cleaned_bp_trans_masked', ''),
                     ('_filtermotart_cleaned_bp', '')
                     ]
   	regex_subs = [('_ts_masker.*/sar', '/smooth/'),
                  ('_ts_masker.*/ar', '/unsmooth/'),
                  ('_combiner.*/sar', '/smooth/'),
                  ('_combiner.*/ar', '/unsmooth/'),
                  ('_aparc_ts.*/sar', '/smooth/'),
                  ('_aparc_ts.*/ar', '/unsmooth/'),
                  ('_getsubcortts.*/sar', '/smooth/'),
                  ('_getsubcortts.*/ar', '/unsmooth/'),
                  ('series/sar', 'series/smooth/'),
                  ('series/ar', 'series/unsmooth/'),
                  ('_inverse_transform./', ''),
                  ]
    	# Save the relevant data into an output directory
    	datasink = Node(interface=DataSink(), name="datasink")
    	datasink.inputs.base_directory = sink_directory
    	datasink.inputs.container = subject_id
    	datasink.inputs.substitutions = substitutions
    	datasink.inputs.regexp_substitutions = regex_subs  # (r'(/_.*(\d+/))', r'/run\2')
    	wf.connect(realign, 'realignment_parameters', datasink, 'resting.qa.motion')
    	wf.connect(art, 'norm_files', datasink, 'resting.qa.art.@norm')
    	wf.connect(art, 'intensity_files', datasink, 'resting.qa.art.@intensity')
    	wf.connect(art, 'outlier_files', datasink, 'resting.qa.art.@outlier_files')
    	wf.connect(registration, 'outputspec.segmentation_files', datasink, 'resting.mask_files')
    	wf.connect(registration, 'outputspec.anat2target', datasink, 'resting.qa.ants')
    	wf.connect(mask, 'mask_file', datasink, 'resting.mask_files.@brainmask')
    	wf.connect(mask_target, 'out_file', datasink, 'resting.mask_files.target')
    	wf.connect(filter1, 'out_f', datasink, 'resting.qa.compmaps.@mc_F')
    	wf.connect(filter1, 'out_pf', datasink, 'resting.qa.compmaps.@mc_pF')
    	wf.connect(filter2, 'out_f', datasink, 'resting.qa.compmaps')
    	wf.connect(filter2, 'out_pf', datasink, 'resting.qa.compmaps.@p')
    	wf.connect(bandpass, 'out_files', datasink, 'resting.timeseries.@bandpassed')
    	wf.connect(smooth, 'smoothed_files', datasink, 'resting.timeseries.@smoothed')
    	wf.connect(createfilter1, 'out_files',
               datasink, 'resting.regress.@regressors')
    	wf.connect(createfilter2, 'out_files',
               datasink, 'resting.regress.@compcorr')
    	wf.connect(maskts, 'out_file', datasink, 'resting.timeseries.target')
    	wf.connect(sampleaparc, 'summary_file',
               datasink, 'resting.parcellations.aparc')
    	wf.connect(sampleaparc, 'avgwf_txt_file',
               datasink, 'resting.parcellations.aparc.@avgwf')
    	wf.connect(ts2txt, 'out_file',
               datasink, 'resting.parcellations.grayo.@subcortical')

    	datasink2 = Node(interface=DataSink(), name="datasink2")
    	datasink2.inputs.base_directory = sink_directory
    	datasink2.inputs.container = subject_id
    	datasink2.inputs.substitutions = substitutions
    	datasink2.inputs.regexp_substitutions = regex_subs  # (r'(/_.*(\d+/))', r'/run\2')
    	wf.connect(combiner, 'out_file',
               datasink2, 'resting.parcellations.grayo.@surface')
    	return wf
	def create_resting_workflow(args, name=None):
    		TR = args.TR
    		slice_times = args.slice_times
    		if args.dicom_file:
        		TR, slice_times, slice_thickness = get_info(args.dicom_file)
        		slice_times = (np.array(slice_times) / 1000.).tolist()
    		if name is None:
        		name = 'resting_' + args.subject_id
    	kwargs = dict(files=[os.path.abspath(filename) for filename in args.files],
                  target_file=os.path.abspath(args.target_file),
                  subject_id=args.subject_id,
                  TR=TR,
                  slice_times=slice_times,
                  vol_fwhm=args.vol_fwhm,
                  surf_fwhm=args.surf_fwhm,
                  norm_threshold=2.,
                  subjects_dir=os.path.abspath(args.fsdir),
                  target_subject=args.target_surfs,
                  lowpass_freq=args.lowpass_freq,
                  highpass_freq=args.highpass_freq,
                  sink_directory=os.path.abspath(args.sink),
                  name=name)
    	wf = create_workflow(**kwargs)
	return wf

if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter
    defstr = ' (default %(default)s)'
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--dicom_file", dest="dicom_file",
                        help="an example dicom file from the resting series")
    parser.add_argument("-f", "--files", dest="files", nargs="+",
                        help="4d nifti files for resting state",
                        required=True)
    parser.add_argument("-t", "--target", dest="target_file",
                        help=("Target in MNI space. Best to use the MindBoggle "
                              "template - "
                              "OASIS-30_Atropos_template_in_MNI152_2mm.nii.gz"),
                        required=True)
    parser.add_argument("-s", "--subject_id", dest="subject_id",
                        help="FreeSurfer subject id", required=True)
    parser.add_argument("--subjects_dir", dest="fsdir",
                        help="FreeSurfer subject directory", required=True)
    parser.add_argument("--target_surfaces", dest="target_surfs", nargs="+",
                        default=['fsaverage5'],
                        help="FreeSurfer target surfaces" + defstr)
    parser.add_argument("--TR", dest="TR", default=None, type=float,
                        help="TR if dicom not provided in seconds")
    parser.add_argument("--slice_times", dest="slice_times", nargs="+",
                        type=float, help="Slice onset times in seconds")
    parser.add_argument('--vol_fwhm', default=6., dest='vol_fwhm',
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument('--surf_fwhm', default=15., dest='surf_fwhm',
                        type=float, help="Spatial FWHM" + defstr)
    parser.add_argument("-l", "--lowpass_freq", dest="lowpass_freq",
                        default=0.1, type=float,
                        help="Low pass frequency (Hz)" + defstr)
    parser.add_argument("-u", "--highpass_freq", dest="highpass_freq",
                        default=0.01, type=float,
                        help="High pass frequency (Hz)" + defstr)
    parser.add_argument("-o", "--output_dir", dest="sink",
                        help="Output directory base", required=True)
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use")
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    args = parser.parse_args()

    wf = create_resting_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.base_dir = work_dir
    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
	wf.run(args.plugin)

