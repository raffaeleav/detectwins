<p align="center">
  <img src="https://github.com/raffaeleav/project-detective/assets/114619463/0fb732f2-4e94-4584-ab40-7d4aa301133e" width="256" heigth="256">
</p>


<p align="center">
  A deepfake detection framework developed as a project for the Fondamenti di Visione Artificiale e Biometria (Fundamentals of Computer Vision and Biometrics) course, part of the Computer Science Master's Degree program at the University of Salerno
</p>


## Table of Contents
- [Authors](#Authors)
- [Overview](#Overview)
- [Features](#Features)
- [Dependencies](#Dependencies)
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


## Dependencies 
- [Artifact dataset](https://github.com/awsaf49/artifact)
- Python dependencies are listed in the "requirements.txt" file


## Built with
- [Python](https://www.python.org/) - used for the Triplet Mining algorithms 
- [Jupyter Notebooks](https://jupyter.org/) - used for model training
