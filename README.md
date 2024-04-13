# RootNav 2

RootNav 2 is a command line tool for the segmentation and analysis of root architectures in 2D. RootNav 2 is still maintained, please reach out if something doesn't work for you.

## Updates 14th April 2024
We are continuing to update RootNav 2 to make using it and training new models as simple as possible. New updates include:
* New logging features, you can use the `--debug` flag when training or analysing images to see much more detail on what is happening
* Code improvements means RootNav 2 runs faster than ever.
* Updated all code to support the latest versions of PyTorch and other libraries, we have also removed the requirement for a few libraries that were no longer needed.
* With this update, the installation should be simpler, with only a few libraries needed to install. We have changed the installation guide to better adhere to the typical ways PyTorch are installed.

## Installing RootNav 2.0 
To install and run rootnav, you will need the following things:
1. A clone of the code from github
2. An installation of python
3. Pytorch and associated libraries
4. Other packages required by the software

If you wish to train your own models, you will also need:
1. An Nvidia GPU (otherwise training will be very slow)
2. Cuda drivers, and pytorch installed with cuda enabled
3. Additional packages required by the software.

The following instructions assume you have installed python, and have compatible hardware if required. If you are not sure how to install python, we recommend using Anaconda, which can be downloaded [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).

## Downloading the RootNav 2.0 Code
You will first need to download the code, either as a zip above, or by cloning the git repository (recommended):
```
git clone https://github.com/robail-yasrab/RootNav-2.0.git
```

## Installing Pytorch
Pytorch is responsible for the deep learning that runs within the Rootnav tool, during both inference and training. Pytorch is updated regularly, and we now recommend installing it following the instructions on the [pytorch website](https://pytorch.org/get-started/locally/).

## Other dependencies
The remaining dependencies can be installed using the requirements files in either the inference or training directories. If you're using pip, then the following will work in Linux:
```
cd RootNav-2.0/inference
pip install -r requirements.txt
```
You can perform the same thing in the training directory, if you need to train new models using RootNav. Library support in other operating systems is more complex, and as above we recommend using [Anaconda](https://docs.anaconda.com/free/miniconda/miniconda-install/). You may find Anaconda is also simplest in Linux as well. 

## Using the tool
The majority of users will want to run RootNav 2.0 on new images, in which case all the code you need is in the `inference` folder. You can find more instructions in the [inference README](https://github.com/robail-yasrab/RootNav-2.0/blob/master/inference/README.md).

### Training new models
Training code may be found in the training folder. Instructions on training models are given in the [training README](https://github.com/robail-yasrab/RootNav-2.0/blob/master/training/README.md). If you would like to collaborate on the development of new models for RootNav 2.0, please contact us.

## Contact
Rootnav 2 is published in GigaScience. For enquiries please contact [michael.pound@nottingham.ac.uk](mailto:michael.pound@nottingham.ac.uk).

## References
<a id="1">[1]</a> 
Yasrab, R., Atkinson, J. A., Wells, D. M., French, A. P., Pridmore, T. P., & Pound, M. P. (2019), 
RootNav 2.0: Deep learning for automatic navigation of complex plant root architectures, 
GigaScience, 8(11), giz123.
