# Inference
The inference code runs the complete Rootnav 2 pipeline. Using a trained network, images are segmented and root tip features located. These are then used by a shortest path search to reconstruct a likely root topology, which is output in the commonly used RSML format.

Our publication releases three trained models on three datasets:
* Wheat grown on blue paper
* Rapeseed grown on blue paper
* Arabidopsis grown on gel plates

The trained models are not included in the repository to save space, instead they are downloaded automatically the first time they are required. New models may appear over time, please do reach out if you have trained on you think would be useful for the community. You can list the currently available models with:
```
cd inference
python rootnav.py --list
```
Currently, the models available are `wheat_bluepaper`, `osr_bluepaper` and `arabidopsis_plate`. To run RootNav 2.0 on a directory of images, use the following command:
```
python rootnav.py --model wheat_bluepaper input_directory output_directory
```
RootNav will read any images within specified input directory, and output images and RSML to the output directory.

In this newer version of the software, the conditional random field (CRF) has been removed. This was found to be more effective during development when the network accuracy was lower. Disabling CRF significantly improves performance for little difference in accuracy. This dependency has also been hard to install on some operating systems.

If you don't have access to a CUDA device, or the relevant drivers you can run rootnav on the CPU using the `--no_cuda` flag processing time per image will be longer, but still <20s per image in our experience.

#### Quantifying Root Systems
RootNav 2.0 doesn't measure root systems itself, it outputs architectures to RSML files, which may be analysed using a suitable tool. We have adapted the original viewer to perform this function, which you can find on github [RootNav Viewer](https://github.com/robail-yasrab/RootNav-Viewer-2.0). The benefit of separating the extraction of root systems and measurement is that additional measurements may be taken at a later date without requiring re-analysis of the original images. This makes analysing and re-analysing many root systems very straightforward. The viewer can also be extended using plugin code to capture new measurements if desired, however most common phenotypes are already built into the tool.
