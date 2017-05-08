# eplilepsy preprocessing in Python

This project tries to preprocess eplilepsy data in a similar way established
preprocessing pipelines, like fcon_1000 and C-PAC, would process their data. 
The preprocessing usually consists of anatomical preprocessing, functional 
preprocessing, registration, segmentation and nuisance signal removal. 

This project has two sub-projects. "pipeline_anjali" utilizes nipype's pipeline
engine and constructs a more complete workflow; "pure_python" is less complete, 
but only depends on pure python packages like nilearn, nipy and pypreprocess. 

## "pipeline_anjali"

some intro

### system requirements

system requirements and how to install dependencies

### usage

how to use


## "pure_python"

This sub-project can apply functional preprocessing and registration on subjects. 
It is pure Python, meaning no external packages (such as FSL and AFNI) needs to
be installed and the code works across platform. 

### system requirements

two packages need direct installation: 
[__nipy__](http://nipy.org/nipy/users/installation.html) and 
[__pypreprocess__](https://github.com/neurospin/pypreprocess#installation). 
Other packages (like nilearn and nibabel) will be installed as parts of the
package dependencies. Note that this project will only require pure Python
parts of pypreprocess, so the instructions on SPM installation can be ignored. 

### usage

To run "everything" 
(i.e. to run functional preprocessing and registration both sequentially
and uninterruptedly), one can type in command line  
```bash
python pipeline_pure_python.py
```
to run with default file locations and settings. To use a more advanced
setting, one can add optional arguments `-dir` to point to "the main
directory holding scans of all subjects", `-sl --subject_list` to point
to the "file listing all subjects", and finally `fc --func_proc_config`
and `-rc --registration_config` to provide links to configuration files
written in .ini format. The parameters stored in these configuration files
will then be passed to function runners to adjust and customize the pipeline. 

### more documentation

check [__here__](http://github.com/anjaligr05/epilepsy_preprocessing/blob/master/README.md)
for detailed workflows, docstrings, and more user guides on topics 
like how to create configuration files, how to create a naming convension, 
how to use parameters and extra paramters, etc. 