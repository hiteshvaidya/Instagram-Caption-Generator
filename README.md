# Instagram-Caption-Generator

This project generates instagram captions for images. It is developed using a combination of Convolutional Neural Network and LSTM Neural Network.

## Prerequisites
You may use [pip](https://pip.pypa.io/en/stable/) or Conda package manager to install these packages
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

## Usage
```python main.py```

## Libraries Used
- Tensorflow
- nltk
- pandas

## Credits
[Yumi's Blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html)
