# RootNav 2.0
This is the RootNav 2.0 Code repository. This readme and the code in the repository is undergoing changes as we submit the work for publication. Instructions on using the training and inference code will be added in due course.

This change will be undergoing refactoring and other useability changes over the coming weeks.

If you are looking simply to use RootNav 2.0 with existing models, you are looking for the inference directory!

### Training
Training code may be found in the training folder.

### Inference
Running the tool using pre-trained models can be done within the inference folder. Three examples are currently provided for the three datasets, a single point of entry and instructions on its use will be added shortly.

### Environment
RootNav 2.0 requires specific libraries in order to run. These include pytorch 1.0.1, torchvision, numpy, yaml and so on. Those who are knowledgeable in python and installing libraries will find it easy to simply run the code, identify missing libraries and install them. For convenience, however, we have provided an anaconda environment file `environment.yml` that will allow you to install all of this automatically. With anaconda installed, run:

`conda env create -f environment.yml`
`conda activate RN2`
