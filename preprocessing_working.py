# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:33:16 2017

@author: chenym
"""

__author__="Anjali Gopal REddy, Yiming Chen"
from nipy import load_image, save_image
from anat_preproc import post_skullstrip
from func_preproc import motion_correction

def main(arg):
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
    if arg == '1': # anatomical preprocessing
        post_skullstrip('defaced_mprage.bse', 'defaced_mprage_brain')
    elif arg == '2': # functional preprocessing
        motion_correction('cube_twist', 'cube_twist_mc_4', 
                          func_dir='cube',mc_alg='SpaceTimeRealign')
if __name__=='__main__':
#	main(sys.argv[1])
    main('2')
