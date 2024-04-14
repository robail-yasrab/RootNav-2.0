# Training
The training code trains new deep networks to simultaneously segment root material in images, and localise key features such as seed locations and tips. Trained models are required as part of the full RootNav 2 pipeline. A broad overview of the process is this:

1. Train a model to the desired accuracy using a training set, periodically checking accuracy against a separate validation set
2. Optionally verify accuracy against a final test set.
3. Publish the trained model weights and a JSON description to the inference folder.
4. Use the inference code to run RootNav 2 on new images.

This readme assumes you have installed the requisite libraries and drivers. A CUDA compatible device is required to train otherwise training will be very slow. The dependencies required can be found in requirements.txt. There are a few additional dependencies beyond those required to run Rootnav 2 in inference mode.

### Training
Instructions on training models are given below. Note that training of deep networks can be a complex task, and experience of the process is recommended. If you would like to collaborate on the development of new models for RootNav 2.0, please [contact us](https://github.com/robail-yasrab/RootNav-2.0#contact).

### Dataset Preparation
RootNav 2 trains on pairs of images and RSML annotations. RSML can be produced by a number of tools, but we use [RootNav 1](https://sourceforge.net/projects/rootnav/) to do this. Exactly how many images you require will depend on the nature of the images, but in our publication we successfully trained to good accuracy using transfer learning on 1-200 images. For very different datasets where transfer learning is harder, more images will be required.

### Dataset Directory Format
The dataset should be split into training and validation sets, and an optional testing set. This follows standard convention for the training of deep networks. As an example, assuming your data is stored in a folder called `new_dataset` your folder structure would be as follows:
```
new_dataset/
    train/
    valid/
    test/ [Optional, used after training]
```

Within each folder should be pairs of images and identically named RSML files. An example dataset is given within the repository [here](https://github.com/robail-yasrab/RootNav-2.0/tree/master/training/OSR_Root_dataset). when training begins the script will scan the directory for valid training pairs, render segmentation masks and store all required training data within cache files in the same directory.

Note that the class weights used by the cross entropy loss are hard coded into the training file. For most datasets these will be satisfactory, you can adapt these to your own dataset if required. We use a script for this [here](https://github.com/robail-yasrab/dataset_weights) and will be adapting this into the training code in due course.

### Configuration files
Training uses a configuration file to store common hyperparameters, along with directory locations and other information. An example may be found within the training code [here](https://github.com/robail-yasrab/RootNav-2.0/tree/master/training/configs). The majority of this is self explanatory and can be left unchanged. You will need to adapt the dataset path to your folder above. You can also specify transfer learning from a previously trained network, if not the network will train from scratch. We recommend transfer learning from wheat_bluepaper as this is the most established network trained for a long time across over 3,000 images.
### Running Training
Training is run using the following command:
```
python training.py train --config ./path/to/config.yml
```
Optionally you can provide the `--output-example` flag to periodically output an RGB image showing an example segmentation each time the network is validated. This may help when checking the progress of training. Using the above command you will see output like this:
```
Iter [50/25000000]  Loss: 0.2852  Time/Image: 0.1760
Iter [100/25000000]  Loss: 0.1300  Time/Image: 0.1682
Iter [150/25000000]  Loss: 0.0905  Time/Image: 0.1689
Iter [200/25000000]  Loss: 0.0760  Time/Image: 0.1693

...
```
Validation results will also appear here when they are run.
### Testing
The training process will save the best performing network within the run/#### folder. This is the best performance on the validation set, rather than the training set. Despite this, a separate test on new data is worthwhile to ensure the network generalises well. The `test` command can be used to run a single iteration over the test set, providing a number of segmentation and localisation metrics in order to measure performance. Testing is run using the following command:
```
python training.py test --config configs/root_train.yml --model ./new_trained_model.pkl
```
As with training, the config file holds the location of the test set, and the number of threads / batch size. Most other configuration requirements are not relevant to testing. You will see output like this:
```
Processed 50 images

Segmentation Results:
Overall Accuracy: 0.9939
Mean Accuracy:    0.7578
FreqW Accuracy:   0.9900
Root Mean IoU:    0.6253

Localisation Results:
Seeds   Precision: 1.0000  Recall: 1.0000  F1: 1.0000
Primary Precision: 0.9442  Recall: 0.9556  F1: 0.9499
Lateral Precision: 0.7237  Recall: 0.8871  F1: 0.7971
```
It should be noted that these results measure the performance of the CNN only, not the full pipeline including path finding. However, a network that increases performance on these metrics is unlikely to perform worse when used within RootNav 2, and so this is a good test to use to compare two trained networks on a dataset.

### Publishing
Training will output network weights as snapshots. RootNav 2 uses these weight files during inference according to settings described in a model JSON configuration. Examples can be found in the inference code. Once training is complete and you have a specific weight file you wish to use with the inference code, the `publish` command can help produce the necessary JSON configuration. The command is used as follows:

```
python training.py publish --name new_dataset_name --model model_weight_file.pt output_dir
```
This will create a default JSON configuration and weight file that may be installed within the `rootnav/inference/models` folder. This configuration file also contains parameters for shortest path searches and other information, which should be adjusted as appropriate for each dataset. Optionally the `--parent name` parameter may be used to specify the parent model (from transfer learning) used during training, e.g. wheat_bluepaper. If a parent is provide, the `--use-parent-config` flag will copy JSON configuration from the parent rather than using default values. Finally, the `--multi-plant` flag specifies that multiple plants are expected during inference. This information is needed for shortest path, not during training.
