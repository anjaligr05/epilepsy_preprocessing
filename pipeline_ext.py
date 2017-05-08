"""Preprocessing pipeline"""
__author__="Anjali Gopal REddy"
''' Function for anatomical preprocessing using nilearn libraries
input args: file (NIFTI) location 
'''
import os
import sys
import numpy as np
import nibabel as nb
from nipype.utils import NUMPY_MMAP

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
	avg = None
	for idx, filename in enumerate(filename_to_list(filename_in)):
        	img = nb.load(filename, mmap=NUMPY_MMAP)
        	data = np.median(img.get_data(), axis=3)
        	if avg is None:
            		avg = data
        	else:
            		avg = avg + data
	median_img = nb.Nifti1Image(avg / float(idx + 1), img.affine,
                                img.header)
	filename = os.path.join(os.getcwd(), 'median.nii.gz')
	median_img.to_filename(filename)
	return filename		
def func_preproc():
	return	
def main(arg):
	#Invoke other functions here
	print("Menu!\n");
	anat_preproc(arg);
	#To be extended, here provide the input files and invoke the functions following
if __name__=='__main__':
	main(sys.argv[1])

