# UNet
Implementation of semantic segmentation model, introduced in a paper titled: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer and Thomas Brox.
```
@misc{ronneberger2015unet,
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation}, 
      author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
      year={2015},
      eprint={1505.04597},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Data
Data comes from [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from Kaggle.  

## Installation
### Clone and install requirements
```bash
$ git clone https://github.com/BigQszak/U-Net.git
$ cd U-Net/
$ pip install requirements.txt
```

## Training 
Edit the hyperparameters.py file to match the setup you want to use. 
Then run train.py with your desired arguments.

## Inference
Run inference.py

## Visualizing output
