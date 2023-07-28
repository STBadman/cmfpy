# Repository for Evaluating Coronal Models CfA Solar REU Project

This repository contains python code to evaluate the three coronal model evaluation metrics defined in [Badman+2022](https://github.com/STBadman/CoronalModelEval/blob/main/references/Badman_2022_ApJ_932_135.pdf), prepared for a 2023 Summer [Solar REU project](https://pweb.cfa.harvard.edu/opportunities/graduate-undergraduate-programs/heassp-solar-physics-reu-program/solar-reu-intern) at the Harvard-Smithsonian Center for Astrophysics. 

* Source code to perform the evaluations for the three metrics are located in subfolders `./CHmetric/`, `./WLmetric/` and `./NLmetric/`, each of which also contains a data folder where outputs are saved. `./helpers.py` contains some general purpose functions used through the project.
* Some example model inputs prepared in the format required by the routines are located in `./example_data/`. 
* `./CHMAP` is a submodule pulling from the Predictive Science Inc. [github repo](https://github.com/predsci/CHMAP/tree/9489ada41fb44c28b2614eab2c6fbd7a834ff870) which provides the ability to perform coronal hole image segmentation via a Fortran bridge. See section [_CHMAP/EZSEG Setup_](https://github.com/STBadman/CoronalModelEval#chmapezseg-setup) below for how to import submodule correctly.
* `./topLevel.ipynb` provides a full run through of the metric implementation applied to one of the example model inputs, explaining the procedure, functions and showing output plots to illustrate the model-data comparisons and where the scores come from.
* `./conda_env.yml` is a conda environment specification file which should install the required dependencies for this notebook and repository to be run out of the box. (6/4/2023 : There is currently a possible issue with downloading EUV data via Sunpy in Windows 10 - see https://github.com/sunpy/sunpy/issues/7045 ). See section [_Environment Setup_](https://github.com/STBadman/CoronalModelEval#environment-setup) for details on how to install the environment. 

## Cloning Repository

To get started, clone this repository to your computer. You will need to have [git]() installed on your computer. Next, navigate (in git-bash (Windows) or Terminal (Mac/Unix/Linux)) to the location on your computer you want to save the repository.

You can either clone via SSH after following the steps in section [Setting Up an SSH Key](https://github.com/STBadman/CoronalModelEval#setting-up-an-ssh-key)  (tested in terminal Unix/Linux)
   
```
git clone git@github.com:STBadman/CoronalModelEval.git
```
   
In Windows, if the ssh step doesn't work, use the https cloning method instead in git-bash :

```
git clone https://github.com/STBadman/CoronalModelEval.git   
```   

## Setting Up an SSH Key 

To clone a repository (including submodules) via SSH, you need to add a public/private SSH key to your GitHub account:
* 1.) <code>cd ~/.ssh</code>
    * navigate to the ssh folder on your computer
* 2.) <code>ls -l</code>
    * check if you already have an ssh key which would be in a file with the <code>.pub</code> extension
* 3.) <code>ssh-keygen -t rsa -b 4096</code>
    * creates public/private SSH key pair saved as a private (`file_name`) and public key file (`file_name.pub`)
* 4.) <code>cat <file_name>.pub</code>
    * view your public key file and copy the text which starts with "ssh-rsa"
* 5.) In your GitHub account, navigate to "Settings" then "SSH and GPG Keys". Then click "New SSH Key" and paste in your SSH key.
    * adds the SSH key to your GitHub account

## Environment Setup

This repository requires a custom environment. Do the following in Anaconda Prompt (windows), Terminal (Mac/Unic/Linux) (or use the GUI in Anaconda Navigator (Windows/Mac)
Check for conda-forge:
```
conda config --show channels
```
If needed, add conda-forge:
```
conda config --append channels conda-forge
```
Now we are ready to build the custom 'coronalmodeleval' conda environment. Navigate to the folder containing the configuration file and run the below snippet.
```
conda env create --file conda_env.yml
```
To activate the conda environment run
```
conda activate coronalmodeleval
```

## CHMAP/EZSEG Setup

In order to set up the Python wrapper for EZSEG do the following in the CHMAP/software/ezseg folder. If you are having errors with import the ezsegwrapper module, make sure to run the following in the EZSEG folder. Additionally, check that the shared module (the ezsegwrapper.so type file name) includes the number for the correct Python version you are running in the filename.

More information on [CHMAP](https://predsci.github.io/CHMAP/), [EZSEG](https://predsci.github.io/CHMAP/chd/chd.html) and the [Fortran wrapper](https://predsci.github.io/CHMAP/chd/f2py.html) can be found in the documentation. 

Setup the CHMAP submodule: 
* 1.) <code>git submodule init</code>
    * initiates the submodule instance found in the [.gitmodules](https://github.com/STBadman/CoronalModelEval/blob/main/.gitmodules) file
* 2.) <code>git submodule update</code>
    * updates and clones the [CHMAP](https://predsci.github.io/CHMAP/) repository

Setup the EZSEG module: This must be done in the `/CHMAP/software/ezseg` directory.
* 1.) <code>cd ./CHMAP/software/ezseg</code>
   * navigate to the EZSEG folder containing the `ezseg.f` file
* 2.) <code>conda activate coronalmodeleval</code>
    * activate conda environment
* 3.) <code>python -m numpy.f2py ezseg.f -m ezsegwrapper -h ezsegwrapper.pyf</code>   
    * creates <code>ezseg.pyf</code> file with ezsegwrapper function 
* 4.) <code>python -m numpy.f2py -c ezsegwrapper.pyf ezseg.f</code>  
    * creates shared module <code>ezsegwrapper.so</code>

## Credits
   
This repository owes substantial credit to efforts of Tamar Ervin, Nicolas Poirier, David Brooks and Harry Warren!!
   
