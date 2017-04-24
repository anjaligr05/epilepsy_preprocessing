__author__="Anjali Gopal Reddy"
""" This is a nilearn based machine learning pipeline. Nilearn comes with code to simplify the use of scikit-learn when dealing with neuroimaging data. I have focussed on funcitonal MRI data 
The following steps were applied before using the machine learning tool
1. Data loading and preprocessing
2. Masking data
3. Resampling images
4. Temporal Filtering and confound removal"""
import sys
import nilearn
def main(arg):
	#Invoke other functions here
	print("Menu!\n");
	#To be extended, here provide the input files and invoke the functions following
if __name__=='__main__':
	main(sys.argv[1])
def preprocessing():
	#downloading the dataset here
	from nilearn import datasets
	dataset = datasets.fetch_haxby()
	list(sorted(dataset.keys())) 
	dataset.func
	print(haxby_dataset['description'])
def load_nonimage():
	import numpy as np
	#behavioral information loading
	behavioral = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
	print(behavioral)
	condition_mask = np.logical_or(conditions == b'face', conditions == b'cat')
	# apply this mask in the sampe direction to restrict the
	# classification to the face vs cat discrimination
	fmri_masked = fmri_masked[condition_mask]

def masking_data():
	from nilearn.input_data import NiftiMasker
	masker = NiftiMasker(mask_img=mask_filename, standardize=True)

	#give the masker a filename and retrieve a 2D array ready
	# for machine learning with scikit-learn
	fmri_masked = masker.fit_transform(fmri_filename)

def learning():
	svc.fit(fmri_masked, conditions)
	###########################################################################
	# predict the labels from the data
	prediction = svc.predict(fmri_masked)
	print(prediction)

def unmasking():
	coef_img = masker.inverse_transform(coef_)
	print(coef_img)

def visualize():
	from nilearn.plotting import plot_stat_map, show

	plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              title="SVM weights", display_mode="yx")

	show()

