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

### Pretrained models
Before training, please download the [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net) pretrained weights (click to download: [u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?pli=1), [u2netp.pth](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view)) and configure the dowloaded path in [config.yml](config.yml).

### training script
We provide the scripts for training our models on the MGD-1K dataset. You can start training with the following commands:
```
python3 train.py models/iter_mask/u2net_mgd1k_itermask_edge_pseudoscribble.py
--gpus=0
--exp-name=pseudo_scribble_mgd1k
--workers=4
```

### Main training function
There are multiple input parameters in the `train.py` file, but most of them have default values.

Input of `train.py`:
```
# These are used to limit the number of clicks during the training of the interactive model.
--num-max-points        # the maximum number of sampled clicks
----max-num-next-clicks      # the maximum number of iteratively sampled clicks each batch

# These are used to set the parameters for the proposed UISM strategy.
--fp-gamma    # upper bound of the model’s uncertain region area     
--fn-gamma    # lower bound of the model’s uncertain region area
--remove-samll-object-area   # remove small target regions from the fp mask or fn mask
```
