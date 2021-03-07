# General Alert!
## New features coming up
![construction image](construction.png)
### I am developing a webpage for this project which will contain additional features like #hash_tag prediction. Sorry for the inconvenience and thank you for visiting this project page

# Instagram-Caption-Generator
This project generates instagram captions for images. It is developed using a combination of Convolutional Neural Network and LSTM Neural Network.

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Design Decisions and Structure](#Design-Decisions-and-Structure)
4. [Pre-requisites](#pre-requisites)
5. [Dataset](#dataset)
6. [Instructions before executing program](#instructions-before-executing-program)
7. [Usage](#usage)
8. [Libraries Used](#libraries-used)
9. [Credits](#credits)

## Introduction
Create a webpage where users could upload their images to be posted on instagram and the webpage predicts a caption based on some captions attached with some of the viral images on instagram. This application is based on Image-to-Text architecture which is a combination of Convolutional Neural Network and Long Short Term Memory (LSTM) network.

Some of the highlights of this project include
* Obtaining features from input images using Convolutional Neural Network 
* Reading image captions from training set in seq-2-seq form in order to generate newer caption text
* Use of Functional API from Tensorflow to develop merge both the networks
* Deploying a deep learning model to production using Flask

## Motivation
I wanted to work on a deep learning project that is not limited to either Computer Vision or Natural Language Processing. With this in mind, I had an option of working on either Visual Question Answering or Caption generation for images. At the same time, an idea struck me that I always wondered what caption should I add for my instagram images and what if there was a system that solves this question for me. Therefore, I thought of working on a Image caption generation problem specifically targetted to pictures posted on Instagram. 

Further, I thought of making a web interface for this system so that people like me who are wondering about a cool instagram caption could get some help. This way, I could learn to make an end-to-end machine learning project that has a wonderful real life application.

## Design Decisions and Structure
* This project reads captions character by character using LSTM. Instead, every caption could be read at once and fed to the LSTM neural network.
* A comparative study of both these approaches would be a nice analysis to perform
* I have used VGG16 CNN in this project simply because of Occam's razor concept. You may try some other network as well. This applies to the configuration of LSTM used as well.

## Pre-requisites
In order to appropriately understand and implement this project, following pre-requisites might come in handy for you,

### Technical pre-requisites
* Familiarity with Tensorflow and Keras
* Working of Convolutional Neural Networks and LSTM networks
* Basics of Natural Language Processing like Word Embeddings

### System pre-requisites
You may use [pip](https://pip.pypa.io/en/stable/) or Conda package manager to install following packages. Simple trick would be to make a `.sh` file and include all the following commands in it and execute that `.sh` file to install all the packages at once.
```bash
pip install tensorflow
pip install nltk
pip install numpy
pip install matplotlib
pip install pandas
```

## Dataset
The dataset required for this program could be downloaded from Kaggle through following steps:
1. Download [Instagram Images with Captions](https://www.kaggle.com/prithvijaunjale/instagram-images-with-captions)
2. Use [Kaggle API](https://github.com/Kaggle/kaggle-api) to download the above mentioned dataset. Once the API is installed, enter following command in the terminal, `kaggle datasets download -d prithvijaunjale/instagram-images-with-captions`
3. In order to read the complete data seamlessly, I made few changes to the directory structure of this dataset which are as follows,
   * Move contents of `data/instagram_data2` to `data/instagram_data`
   * Add column names from `data/instagram_data/captions_csv2.csv` viz. 'Image File' and 'Caption' to `data/instagram_data/captions_csv2.csv`
   * The readData.py script will now read data from both `.csv` files and merge all the images

## Instructions before executing program
1. Please create a new directory named `output` in the project directory
2. You may change program hyperparameters like number of epochs and image size in `config.py`

## Usage
```python main.py```

## Libraries Used
- Tensorflow
- Keras
- nltk
- pandas
- Scikit-learn

## Credits
[Yumi's Blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html)
