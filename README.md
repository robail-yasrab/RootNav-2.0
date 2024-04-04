# RootNav 2.1
This is the RootNav 2.1 Code Branch. This README and the code in this branch is being improved before being pulled back to the main code repository. Here we are trying to implement improvements dependencies, including making install easier.

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

The following instructions assume you have installed python, and have compatible hardware if required.

## Downloading the RootNav 2.0 Code
You will first need to download the code, either as a zip above, or by cloning the git repository (recommended):
```
git clone https://github.com/robail-yasrab/RootNav-2.0.git
```

## Installing Pytorch
Pytorch is responsible for the deep learnin that runs within the Rootnav tool, during both inference and training. Pytorch is updated regularly, and we recommend installing it following the instructions on the pytorch website:

https://pytorch.org/get-started/locally/

## Other dependencies
The remaining dependencies can be installed using the requirements files in either the inference or training directories. If you're using pip, then the following will work in Linux:
```
cd RootNav-2.0/inference
pip install -r requirements.txt
```
You can perform the same thing in the training directory, if you need to train new models using RootNav. Library support in other Operating Systems is more complex, and we recommend using [Anaconda](https://www.anaconda.com/). You may find Anaconda is also simplest in Linux as well. 

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
