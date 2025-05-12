# DeepFRC update on 2025/5/12

## Introduction

DeepFRC is an end-to-end deep learning framework for joint functional registration and classification.

![DeepFRC Model](images/The_Model.png)

Overview of DeepFRC. The light blue region highlights the 1D convolutional module. The three blue arrows (left to right) denote: flattening of the $(l_1{-}1)$ th layer matrix, the time warping (TW) operation, and the basis projection of the aligned function. ``TW`` and ``DR`` denote time warping and dimensionality reduction, respectively. We set $l_1 =4, l_2=3$.

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
