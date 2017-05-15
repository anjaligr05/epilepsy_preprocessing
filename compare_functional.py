# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 01:40:47 2017

@author: chenym
"""

import re
import matplotlib.pyplot as plt
from nipy import load_image
import numpy as np

if __name__ == '__main__': # main

    rest = load_image('func/rest.nii.gz').get_data()
    afni = load_image('sample/rest_mc.nii.gz').get_data()
    pypr = load_image('func/rest_mc.nii.gz').get_data()
    nipy = load_image('func/rest_mc_2.nii.gz').get_data()

#    groups = {'raw':raw, 'sr':sr, 'pyp':pyp}
#    control = original
#    control_name = 'original_cube'
    groups = {'nipy':nipy, 'pypr':pypr, 'afni':afni}
    control = rest
    control_name = 'rest'
    dim_x, dim_y, dim_z, dim_t = control.shape

    while True:
        inp = raw_input('put in three seperated integers (e.g., 2,40,25) '+\
        'as coordinates; "s" for summary; "q" to exit. '+\
        'Dimension: %s\n' % str(control.shape[0:-1]))
        if inp == 'q':
            break
        if inp == 's':
            plt.figure()
            plt.title('error per voxel per average voxel intensity, for each time slice')
            t = range(dim_t) # how many timesteps
            for comp_name in groups.keys():
                tbc = groups[comp_name].astype('float') # cast to Long
                ori = control.astype('float') # to avoid overflow
                stat = np.sum(np.abs(tbc-ori),axis=(0,1,2))/\
                np.sum(ori,axis=(0,1,2))
                avg = np.sum(np.abs(tbc-ori))/np.sum(ori)
                print 'average error for %s is %f' % (comp_name, avg)
                avg = avg*np.ones_like(t)
                # plots
                plt.plot(t,stat,label='errors for '+comp_name)
                plt.draw()
                plt.plot(t,avg,label='avg for '+comp_name)
                plt.draw()
            plt.legend(loc='upper right')
            plt.draw()
            continue
            
            
        mch = re.match('[ \[\(]*(\d+)[ ,]+(\d+)[ ,]+(\d+)[ ,\]\)]*', inp)
        if not mch:
            print 'invalid sytax, try again.'
            continue
        x,y,z = [int(numstr) for numstr in mch.groups()]
        print('voxel of choice: (x=%d,y=%d,z=%d); '% (x,y,z) +\
        'showing: time series graphs of three mcs. ')
        
        t = range(dim_t) # how many timesteps
        plt.figure()
#        plt.subplots()
        for comp_name in groups.keys():
            plt.plot(t,groups[comp_name][x,y,z,:],label=comp_name)
            plt.draw()
        plt.plot(t,control[x,y,z,:],label=control_name)
        plt.draw()
        plt.legend(loc='upper center')
        plt.draw()
#        plt.show()
        
        