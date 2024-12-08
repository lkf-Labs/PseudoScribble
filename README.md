# PseudoScribble
## PseudoScribble: Interactive Meibomian Gland Segmentation with Scribble Simulation

## Environment
Training and evaluation environment: Python3.9.4, PyTorch 2.3.1, CUDA 12.4. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```
## Data
We train and evaluate all our models on our internal private dataset MG-203 and the public dataset MGD-1K. The [MGD-1K](https://mgd1k.github.io/) dataset can be accessed online.

We assume the data folder (`data_dir`) has the following structure:

```
datasets
├── <dataset_name> 
│ └── img
│   └── ...
│ └── label
│   └── ...
│ └── train
│   └── train.txt
│ └── val
│   └── val.txt
│ └── test
│   └── test.txt
│ └── metrics
```

## Training
Before training, please download the [$U^2-Net$](https://github.com/xuebinqin/U-2-Net) pretrained weights (click to download: [u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?pli=1), [u2netp.pth](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view)) and configure the dowloaded path in [config.yml](https://github.com/uncbiag/SimpleClick/blob/main/config.yml).
