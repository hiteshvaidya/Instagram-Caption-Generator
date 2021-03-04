# Instagram-Caption-Generator

This project generates instagram captions for images. It is developed using a combination of Convolutional Neural Network and LSTM Neural Network.

## New features coming up
![construction image](https://lh3.googleusercontent.com/proxy/gCRrLDAr00rADvOBSqO6aCeDzJe4FKZMmRi5qJ2e36fR5sLji0v-o-G2lW0VoBSIONs3Mw37UlVai8lg5RbJWDUU6EG9PTbWhX2DkiOq0SsU6eHImYvvHLlS)
### I am developing a webpage for this project which will contain additional features like #hash_tag prediction. Sorry for the inconvenience and thank you for visiting this project page

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
3. In order to read the complete data seamlessly, I made few changes to the directory structure of this dataset which are as follows,
  * Move contents of `data/instagram_data2` to `data/instagram_data`
  * Add column names from `data/instagram_data/captions_csv2.csv` viz. 'Image File' and 'Caption' to `data/instagram_data/captions_csv2.csv`
  * The readData.py script will now read data from both `.csv` files and merge all the images

## Usage
```python main.py```

## Libraries Used
- Tensorflow
- nltk
- pandas

## Credits
[Yumi's Blog](https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html)
