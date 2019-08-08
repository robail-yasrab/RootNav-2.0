## RootNav 2.0 Changelog

### 8 August 2019 - New features
* CRF support is now optional, allowing those that have trouble installing this requisite to skip it.
* You can now look at information on a trained model using `python rootnav.py --info modelname`. 
* Models now include more detailed information on the trainer, and owners of the dataset used.
* Models also include information on which networks were used for transfer learning, if used.
* Training code now uses a simpler configuration file, we are continuing work on this.

### 29 July 2019 - New features
* Python 3.6 support. This should make it much easier to install on Windows, for example. It's also made RootNav 2.0 about 30% faster!
* Simplified environment and requirement files. Some unecessary dependencies have been removed to speed up and simplify installation.
* CPU only support, RootNav 2.0 will now run on a laptop and still pretty quickly.
* Automatic download of trained models.
* Command line parameter support to list and select models, as well as input and output directories