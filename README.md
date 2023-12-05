<<<<<<< HEAD
# KiTS-SAM-Adapter

## Prerequisites
- Linux (We tested our codes on Ubuntu 18.04)
- Anaconda
- Python 3.10.11
- Pytorch 2.0.0 **(Pytorch 2+ is necessary)**

To get started, first please clone the repo
```
git clone https://github.com/StevenMA616777/KiTS-SAM-Adapter.git
```
Then, please run the following commands:
```
conda create -n KiTS23 python=3.10.11
conda activate KiTS23
pip install -r requirements.txt
```

## Quick start
You need to prepare the [vit_h version of SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Training
We adopt one A100 (80G) for training.
1. Prepare your dataset and convert the nii.gz image into png format. Save the names in the list file in './list' folder.
2. Run this command to train SAMed.
```bash
python train.py --root_path <Your folder> --output <Your output path> --warmup --AdamW --tf32 --compile --use_amp --lr_exp 7 --max_epochs 400 --stop_epoch 300
```
Check the results in `<Your output path>`, and the training process will consume about 70G GPU memory.
