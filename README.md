# LARNet: Latent Action Representation for Human Action Synthesis
Code repository for BMVC paper LARNet: Latent Action Representation for Human Action Synthesis
Paper available at [arxiv] (https://arxiv.org/pdf/2110.10899.pdf)

Project page with more information available in [CRCV webpage] (https://www.crcv.ucf.edu/research/projects/larnet-latent-action-representation-for-human-action-synthesis)

Demo available [HERE](https://www.crcv.ucf.edu/wp-content/uploads/2018/11/LARNet_BMVC21_demo.mp4)

![](https://github.com/aayushjr/larnet/blob/main/res/NTU.gif)

![](https://github.com/aayushjr/larnet/blob/main/res/KTH.gif)

![](https://github.com/aayushjr/larnet/blob/main/res/UTD.gif)

![](https://github.com/aayushjr/larnet/blob/main/res/Penn.gif)

![](https://github.com/aayushjr/larnet/blob/main/res/Syn.gif)

## Description
This is an implementation of LARNet on NTU-RGB+D 60 dataset. It is built using the PyTorch library. The datasets have to be downloaded separately.

### Setup

Please install following prerequisites: Python>=3.6, Pytorch>=1.6

Install remainig libraries from requirements.txt using 
```
pip install -r requirements.txt
```

Download datasets and set the `data_root_dir` variable in `train.py` for each dataset.

Download I3D pretrained weights for Charades / Kinetics and setup the path in `i3d_weights_path` variable in `train.py`.

If you are resuming training, set the weights for generator and discriminator and uncomment weight load command for each in `train.py`.

### Training

After doing setup and putting data/model path correctly in train.py, run using:
```
python train.py
```

If you enable tensorboard, it will save the generated video for each snapshot interval in the tensorboard log file.