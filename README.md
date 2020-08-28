# RootNav 2.0
This is the RootNav 2.0 Code repository. This README and the repository code are being improved daily as we prepare for publication, please check back for new features and documentation soon!

### 24th August 2020 - Version 2.1
Version 2.1 makes substantial improvements to the training code for RootNav 2, making it easier to train the tool on new datasets. Features include:
* Simplified folder structure, simply pass `train` and `valid` folders that contain images and RSML descriptions of the root systems.
* Masks for training are now created by default when training begins, and cached for future runs. Only RSML is required.
* Once training is complete, you can publish a model to be used within RootNav 2 using the `training.py publish` function.

### Longer term features
* Additional trained models. If you have datasets you'd like to see work with the tool and aren't covered by our examples, please get in touch and we can collaborate!
* Improved model and training code. We will continuously review the model architectures and training code to explore ways to increase performance.

## Installing RootNav 2.0 
You will first need to download the code, either as a zip above, or by cloning the git repository (recommended):
```
git clone https://github.com/robail-yasrab/RootNav-2.0.git
```
Next, install the required dependencies. If you're using pip, then the following will work in Linux:
```
cd RootNav-2.0
pip install -r requirements.txt
```
Library support in other Operating Systems is more complex, and we recommend using [Anaconda](https://www.anaconda.com/). You may find Anaconda is also simplest in Linux as well. First install anaconda, then create a new environment using the included yml dependencies file.
```
conda env create -f environment.yml
conda activate rootnav2
```
This will download and set up all the required libraries, providing you with a Python 3.6 installation. For those with access to Python 2.6 only, you will find a compatible version in the [py2 branch](https://github.com/robail-yasrab/RootNav-2.0/tree/py2).

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
