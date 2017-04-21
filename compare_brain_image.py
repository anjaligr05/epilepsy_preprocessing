# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 17:33:05 2017

@author: chenym
"""
from nipy import load_image
import re
import matplotlib.pyplot as plt

arbint = 15
vminval = -2
vmaxval = 2

class BrainImageComparator:
    
    @classmethod
    def create_from_file(cls, *args):
        images = {}
        for file_name in args:
            # filename such as /.../.../rest.nii.gz; remove /.../.../ parts
            file_namer_no_directory = file_name.split('/')[-1]
            # extract rest from rest.nii.gz
            ex = re.findall('\S+(?=\.nii\.gz)',file_namer_no_directory)
            if len(ex) is not 1:
                raise ValueError('argument "'+file_name+\
                '" is not an acceptable brain imaging file name. ')
            name_key = ex[0]
            img_value = load_image(file_name)
            images[name_key] = img_value # collect in a dict for later use
        
        return cls(images)
    
    def __init__(self,image_dict):
        self.images = image_dict
        
    def _plot(self, name, data, arbint, save_no_view=False, save_dir=None):
        plt.figure()
        plt.imshow(data[:,:,arbint], vmin=vminval, vmax=vmaxval)
        plt.title('cross section of ' + name + ', at z = '+ str(arbint))
        plt.colorbar()
        if save_no_view:
            plt.savefig(save_dir+re.sub('\W+','_',name)+'_'+str(arbint)+'.png')
            plt.close()
            
    def compare(self, arbints, save_no_view=False, save_dir=None):

        if len(self.images) < 2:
            raise RuntimeError('need at least two images to compare. ')
        
#        report = 'checking dimensions...'
#        standard = None
#        for name in self.images.keys():
#            if standard is None:
#                standard = self.images[name].shape
#                continue
#            if standard != self.images[name].shape:
#                report += '\ndimension not matched. \n'
#                return report
#        report += '\ndimension of all images: \n' + str(standard)
#        
#        report += 'checking affine transformation matrices...'

        for arbint in arbints:
            namelist = self.images.keys()
            for name in namelist:
                data = self.images[name].get_data()
                self._plot(name,data,arbint,save_no_view,save_dir)
            
            while len(namelist) > 1:
                name = namelist.pop()
                for othername in namelist:
                    data = self.images[name].get_data() \
                    - self.images[othername].get_data()
                    self._plot('the difference between '+name+' and '+othername,\
                    data,arbint,save_no_view,save_dir)

#comp = BrainImageComparator('C:/Users/chenym/OneDrive/Documents/DR/brainsuite/defaced_mprage_2.nii.gz',\
#'C:/Users/chenym/OneDrive/Documents/DR/brainsuite/defaced_mprage_mask.nii.gz')

#comp = BrainImageComparator('samples/defaced_mprage.nii.gz',\
#'samples/defaced_mprage_surf.nii.gz','samples/defaced_mprage_brain.nii.gz',\
#'temp/defaced_mprage_brain2.nii.gz')
#comp = BrainImageComparator.create_from_file('samples/defaced_mprage_brain.nii.gz',\
#'temp/defaced_mprage_brain2.nii.gz')
#comp.compare(range(0,156,6),save_no_view=True,save_dir='C:/Users/chenym/Downloads/imgtemp/')

standard = load_image('cube/cube.nii.gz')
twist = load_image('cube/cube_twist.nii.gz')
mc = load_image('cube/cube_twist_mc.nii.gz')
mc2 = load_image('cube/cube_twist_mc_2.nii.gz')
#mc2 = load_image('temp/rest_mc2.nii.gz')
comp = BrainImageComparator(\
{'standard':standard[:,:,arbint,:],\
'twist':twist[:,:,arbint,:],\
'mc':mc[:,:,arbint,:],\
'mc2':mc2[:,:,arbint,:]})
comp.compare(range(0,150,15),save_no_view=True,save_dir='C:/Users/chenym/Downloads/imgtemp/')