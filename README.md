# DeepFRC update on 2025/9/26

## Introduction

DeepFRC is an end-to-end deep learning framework for joint functional registration and classification.

![DeepFRC Model](images/overview.png)

Overview of DeepFRC. Multiple raw functional trajectories are first aligned via a neural deformation operator (1D CNN) that learns diffeomorphic time warping $\gamma(t)$, producing both warped curves and alignment functions. The aligned signals are then expanded in a Fourier basis to obtain spectral coefficients $c_1, \dots, c_K$, which serve as inputs to a classifier network (MLP with Softmax output) for predicting class labels $y \in {1,\dots,C}$. Training is performed jointly by minimizing a contrastive geometric alignment loss and a classification loss.

The Paper Link: 

## Getting Started

The code running order:

1# model/data/data_gen.ipynb

2# model/data/data_preprocess.ipynb

3# model/DeepFRC.ipynb

## Dataset Download

Wave (motion): https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryX

Yoga (image-derived): https://www.timeseriesclassification.com/description.php?Dataset=Yoga

Symbol (handwriting): https://www.timeseriesclassification.com/description.php?Dataset=Symbols





