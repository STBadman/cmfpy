# Repository for Evaluating Coronal Models CfA Solar REU Project

## Environment Setup

This requires a custom environment.
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

In order to do this, you need to add a public/private SSH key to your GitHub account:
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
