# RootNav 2.0
This is the RootNav 2.0 Code repository. This readme and the code in the repository is undergoing changes as we submit the work for publication. Instructions on using the training and inference code will be added in due course.

This change will be undergoing refactoring and other useability changes over the coming weeks.

If you are looking simply to use RootNav 2.0 with existing models, you are looking for the inference directory!

## Update 26 Jul 2019
Many thanks for all the interest in this repository! Since our preprint was released and the paper submitted for review, we are working hard on usability and code improvements to make sure the tool is as accessible as possible. These changes should only take a week or so longer, and we will continue to make more improvements into the future! Please watch this repo for updates very soon.

### Upcoming Features
* Much simplified environment and requirement files, making installing the tool easier
* Code refactoring so that it is easier to run and train new models
* CPU only support for those running the tool without an Nvidia graphics card and cuda installed (it's actually not much slower for running rootnav 2, training still requires a GPU)
* Automatic download of trained models if the weight files are absent (they are not included in github by default due to their size)
* More documentation!

### Longer term features
* Additional trained models. If you have datasets you'd like to see work with the tool and aren't covered by our examples, please get in touch and we can c

## Using the tool

### Training
Training code may be found in the training folder.

### Inference
Running the tool using pre-trained models can be done within the inference folder. Three examples are currently provided for the three datasets, a single point of entry and instructions on its use will be added shortly.

### Environment
RootNav 2.0 requires specific libraries in order to run. These include pytorch 1.0.1, torchvision, numpy, yaml and so on. Those who are knowledgeable in python and installing libraries will find it easy to simply run the code, identify missing libraries and install them via pip. For convenience, however, we have provided an anaconda environment file `environment.yml` that will allow you to install all of this automatically. With anaconda installed, run:

`conda env create -f environment.yml`
`conda activate RN2`

We will be simplifying this process as a priority, so please check back in a few days.