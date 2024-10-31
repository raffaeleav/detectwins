<p align="center">
  <img src="https://github.com/user-attachments/assets/df6d4f56-eca2-4f57-8b60-7f82823f40c8" width="512" heigth="120">
</p>


<p align="center">
  A deepfake detection framework developed as a project for the Fondamenti di Visione Artificiale e Biometria (Fundamentals of Computer Vision and Biometrics) course, part of the Computer Science Master's Degree program at the University of Salerno
</p>


## Table of Contents
- [Authors](#Authors)
- [Overview](#Overview)
- [Features](#Features)
- [Results](#Results)
- [Requirements](#Requirements)
- [How to replicate](#How-to-replicate)
- [Built with](#Built-with)


## Authors 
| Name | GitHub profile |
| ------------- | ------------- |
| Aviello Raffaele  | [raffaeleav](https://github.com/raffaeleav) |
| Califano Alfonso | [Ackermann32](https://github.com/Ackermann32) |
| Capuano Rachele | [raaacheeelc](https://github.com/raaacheeelc) |


## Overview 
  Detectwins was developed to explore Triplet Mining (and its variations) for deepfake detection while using Fourier-transformed 
  images. The latter technique is used to leverage frequency 
	domain representation of images, where visual signals are converted to highlight details of the images that are not visible in 
 the spatial domain (e.g. ai artifacts). The framework is also 
	capable of analyzing RGB images (with some success).


## Features
1) Offline Semi-Hard Mining
2) Offline Hard Mining
3) Online Hard Mining
4) One-Shot Learning


## Results
| Method | Accuracy | Precision | Recall | Specifity | F1 Score |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| Online Hard Mining | 88.10 | 80.65 | 84.60 | 89.85 | 82.57 |
| One Shot Learning (BigGan) | 90.93 | 79.59 | 97.89 | 87.45 | 86.67 |
| One Shot Learning (Latent Diffusion) | 78.36 | 69.35 | 62.90 | 86.10 | 65.71 |


## Requirements 
- [Artifact dataset](https://github.com/awsaf49/artifact)
- Python dependencies are listed in the "requirements.txt" file


## How to replicate
1) Clone the repository
```bash
git clone https://github.com/raffaeleav/detectwins.git
```
2) Install dependencies (assuming conda is being used)
```bash
conda create -n "detectwins" python=3.10 
conda activate detectwins
pip install -r detectwins/requirements.txt
```
3) Install Cuda toolkit (assuming you have Ubuntu 22.04)
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.runsudo
sh cuda_12.1.0_530.30.02_linux.run
pip install torch torchvision torchaudio
```
4) Download Artifact dataset
```bash
kaggle datasets download -d awsaf49/artifact-dataset
unzip artifact-dataset.zip -d artifact
```
5) Switch to the project directory
```bash
cd detectwins
```
6) Start the Jupyter server and run testing 
```bash
jupyter notebook notebooks/testing.ipynb
```


## Built with
- [Python](https://www.python.org/) - used for the Triplet Mining algorithms 
- [Jupyter Notebooks](https://jupyter.org/) - used for model training
