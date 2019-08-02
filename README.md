# RootNav 2.0
This is the RootNav 2.0 Code repository. This README and the repository code are being improved daily as we prepare for publication, please check back for new features and documentation soon!
<p align="center">
<a href="https://www.youtube.com/watch?v=IJQurx2Az2w" target="_blank">
<a href="https://www.youtube.com/watch?v=Hab2Xtk7LPs" target="_blank"> 
</p>

### 29 July 2019 - New features
* Python 3.6 support. This should make it much easier to install on Windows, for example. It's also made RootNav 2.0 about 30% faster!
* Simplified environment and requirement files. Some unecessary dependencies have been removed to speed up and simplify installation.
* CPU only support, RootNav 2.0 will now run on a laptop and still pretty quickly.
* Automatic download of trained models.
* Command line parameter support to list and select models, as well as input and output directories

### Upcoming Features
* Extended documentation to make clear how the tool is used

### Longer term features
* Additional trained models. If you have datasets you'd like to see work with the tool and aren't covered by our examples, please get in touch and we can collaborate!

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
This will download and set up all the required libraries, providing you with a Python 3.6 installation. For those with access to Python 2.6 only, you will find a compatible version in the `py2` branch.

## Using the tool
The majority of users will want to run RootNav 2.0 on new images, in which case all the code you need is in the `inference` folder.

### Inference
Running the tool using pre-trained models may be done within the inference folder. Our publication releases three trained models on three datasets:
* Wheat grown on blue paper
* Rapeseed grown on blue paper
* Arabidopsis grown on gel plates

The trained models are not included in the repository, instead they are downloaded automatically the first time they are required. You can list the currently available models with:
```
cd inference
python rootnav.py --list
```
Currently, the models available are `wheat_bluepaper`, `osr_bluepaper` and `arabidopsis_plate`. To run RootNav 2.0 on a directory of images, use the following command:
```
python rootnav.py --model wheat_bluepaper input_directory output_directory
```
RootNav will read any images within specified input directory, and output images and RSML to the output directory.

### Training
Training code may be found in the training folder. Instructions on training models are given below. If you would like to collaborate on the development of new models for RootNav 2.0, please contact us.


### Dataset Directory Format
Please follow the bellow given directory formate for the training process. 
```
- Roots
  -- train  (training dataset images)
  -- trainannot (RGB ground truth images)
  -- val (validation dataset images)
  -- valannot (RGB validation ground truth images)
  -- RSML (RSML annotation)
```  
  
### Usage
**Setup config file**
```yaml
# Model Configuration
model:
    arch: <name> [options: 'Hourglass, Nested_HG:<value>

# Data Configuration
data:
    dataset: <name> [options: 'wheat_roots', OSR_roots] 
    train_split: <split_to_train_on>
    val_split: <spit_to_validate_on>
    img_rows: 512
    img_cols: 1024
    path: <path/to/dataset>


# Training Configuration
training:
    n_workers: 64
    train_iters: 35000
    batch_size: 16
    val_interval: 500
    print_interval: 25
    loss:
        name: <loss_type> [options: 'cross_entropy, bootstrapped_cross_entropy, multi_scale_crossentropy']
        <loss_keyarg1>:<value>

    # Optmizer Configuration
    optimizer:
        name: <optimizer_name> [options: 'sgd, adam, adamax, asgd, adadelta, adagrad, rmsprop']
        lr: 1.0e-3
        weight_decay: 0.00005
        momentum: 0
        
    # Augmentations Configuration
    augmentations:
        rotate: d                                    #[rotate -d to d degrees]
        hflip: p                                     #[flip horizontally with chance p]


    # LR Schedule Configuration
    lr_schedule:
        name: <schedule_type> [options: 'constant_lr, poly_lr, multi_step, cosine_annealing, exp_lr']
        <scheduler_keyarg1>:<value>

    # Transfer Leaning ... path to pre-trained model  
    # Resume from checkpoint  
    resume: <path_to_checkpoint>
```
**Model Training:**

```
python train.py [-h] [--config [CONFIG]] 

--config                Configuration file to use

# example: python train.py --config ./configs/rootnav2_Nested_HG.yml 
```

## Contact
Publication details will appear here in due course. For enquiries please contact [michael.pound@nottingham.ac.uk](mailto:michael.pound@nottingham.ac.uk).
