'''
This file contains code for generating captions for a given images
Version: 1.0
Author: Hitesh Vaidya
'''

# import libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split
from metadata.readData import clean_data, load_images_list
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from metadata import config
from collections import OrderedDict
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_features(df):
    '''
    Extract features from each photo in the directory
    :param directory: directory location
    :return: features
    '''

    # load the model
    model_vgg = tf.keras.applications.VGG16(include_top=True,
                                            weights='imagenet')
    # load locally saved weights
    # model.load_weights('<locally saved weights>')
    print('VGG16 summary:')
    # pop the last layer of VGG16 model since we need only the features and
    # not final recognition labels
    model_vgg.layers.pop()
    model_vgg = Model(inputs=model_vgg.inputs, outputs=model_vgg.layers[
        -1].output)
    # Show the summary of model
    print(model_vgg.summary())

    images = OrderedDict()
    # image size is fixed at 224 because VGG16 model has been pre-trained to take
    # that size

    img_path = config.BASE_INPUT
    images_list = df['Image File']
    # data = np.zeros((len(images_list), config.npix, config.npix, 3))
    tqdm.write('Reading images and obtaining their features')
    for i, name in tqdm(enumerate(images_list)):
        # load an image from file
        filename = os.path.join(os.getcwd(), img_path, name+'.jpg')
        # load image in PIL format
        image = load_img(filename, target_size=config.target_size)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # preprocess image to be compliant with pretrained VGG16 in keras
        nimage = tf.keras.applications.vgg16.preprocess_input(image)
        # reshape nimage as (batch_size=1, nimage_shape)
        y_pred = model_vgg.predict(nimage.reshape((1,) + nimage.shape[:3]))
        images[name] = y_pred.flatten()

    return images


def preprocessing(dtexts, dimages):
    '''
    Pre-process the captions to generate seq2seq text
    Caption is generated upto position 't' and the model learns to predict
    caption at 't+1' position.
    :param dtexts: captions
    :param dimages: image features
    :return:    Xtext - input text upto point 't'
                ytext - text to be predicted at 't+1' position
                Ximage - image features
    '''
    N = len(dtexts)
    print("# captions/images = {}".format(N))

    # assert error if number of captions and images do not match
    assert (N == len(dimages))

    Xtext, Ximage, ytext = [], [], []
    for text, image in zip(dtexts, dimages):
        for t in range(1, len(text)):
            in_text, out_text = text[:t], text[t]
            in_text = pad_sequences([in_text], maxlen=config.maxlen).flatten()
            out_text = to_categorical(out_text, num_classes=config.vocab_size)

            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)

    Xtext = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext = np.array(ytext)
    print("Xtext: {}, Ximage: {}, ytext: {}".format(Xtext.shape, Ximage.shape,
                                                    ytext.shape))
    return (Xtext, Ximage, ytext)


def img_to_text_model(input_shape):
    '''
    Final model that combines image features from VGG16 and their
    respective captions
    :param input_shape: size of image features
    :return:
    '''
    embedding_dimension = 64

    # Input layer
    input_image = layers.Input(shape=(input_shape,))
    fimage = layers.Dense(256, activation='relu', name="ImageFeature")(
        input_image)

    # sequence model
    input_txt = layers.Input(shape=(config.maxlen,))
    ftxt = layers.Embedding(config.vocab_size, embedding_dimension,
                            mask_zero=True)(input_txt)
    ftxt = layers.LSTM(256, name="CaptionFeature")(ftxt)

    # combined model for decoder
    decoder = layers.add([ftxt, fimage])
    decoder = layers.Dense(256, activation='relu')(decoder)
    output = layers.Dense(config.vocab_size, activation='softmax')(decoder)

    # define the model
    model = Model(inputs=[input_image, input_txt], outputs=output)
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print('Image to text model summary')
    print(model.summary())
    return model


def metric(hist):
    '''
    graphs of metrics
    :param hist: model
    :return: None
    '''
    for label in ['loss', 'val_loss']:
        plt.plot(hist.history[label], label=label)
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def predict_caption(img_text_model, image, tokenizer, index_word):
    '''
    image.shape = (1,4462)
    :param index_word: inverse of tokenizer
    :param img_text_model: final neural network model
    :param tokenizer: tokenizer
    :param image: image feature vector
    :return: predicted caption
    '''
    in_text = '<START>'

    for iword in range(config.maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], config.maxlen)
        yhat = img_text_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        new_word = index_word[yhat]
        in_text += ' ' + new_word
        if new_word == '<END>':
            break
    return (in_text)


def plot_images(img_text_model, fnm_test, di_test, tokenizer, index_word):
    '''
    plots images and their predicted captions
    :param img_text_model: final neural network model
    :param fnm_test: image filenames from test set
    :param di_test: image feature vector from test set obtained from VGG16 model
    :param tokenizer: tokenizer
    :param index_word: inverse of tokenizer
    :return: None
    '''
    npic = 5
    npix = 224
    target_size = (npix, npix, 3)

    count = 1
    fig = plt.figure(figsize=(10, 20))
    for jpgfnm, image_feature in zip(fnm_test[:npic], di_test[:npic]):
        # images
        filename = os.path.join(config.BASE_INPUT, jpgfnm)
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
        ax.imshow(image_load)
        count += 1

        # caption
        caption = predict_caption(img_text_model,
                                  image_feature.reshape(1, len(image_feature)),
                                  tokenizer,
                                  index_word)
        ax = fig.add_subplot(npic, 2, count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0, 0.5, caption, fontsize=20)
        count += 1

    plt.show()


def bleu_score(img_text_model, fnm_test, di_test, dt_test, tokenizer,
               index_word):
    '''
    Calculates BLEU score
    :param img_text_model: final neural network model
    :param fnm_test: filename of image from test set
    :param di_test: image feature vector in test set obtained from VGG16
    :param dt_test: caption from test set
    :param tokenizer: tokenizer
    :param index_word: inverse of tokenizer
    :return: None
    '''
    # number of final results to be stored
    nkeep = 5
    # good and bad predictions and bleu scores
    pred_good, pred_bad, bleus = [], [], []
    count = 0
    for jpgfnm, image_feature, tokenized_text in zip(fnm_test, di_test,
                                                     dt_test):
        count += 1
        if count % 200 == 0:
            print("  {:4.2f}% is done..".format(
                100 * count / float(len(fnm_test))))

        caption_true = [index_word[i] for i in tokenized_text]
        caption_true = caption_true[1:-1]  ## remove <START>, and <END>>
        # captions
        caption = predict_caption(img_text_model,
                                  image_feature.reshape(1, len(image_feature)),
                                  tokenizer,
                                  index_word)
        caption = caption.split()
        caption = caption[1:-1]  ## remove startreg, and endreg

        bleu = sentence_bleu([caption_true], caption)
        bleus.append(bleu)
        if bleu > 0.7 and len(pred_good) < nkeep:
            pred_good.append((bleu, jpgfnm, caption_true, caption))
        elif bleu < 0.3 and len(pred_bad) < nkeep:
            pred_bad.append((bleu, jpgfnm, caption_true, caption))

    print('Mean BLEU {:4.3f}'.format(np.mean(bleus)))


def main():
    '''
    Main function
    :return: None
    '''

    # obtain a cleaned data frame
    df = clean_data()
    # extract feature vectors of image from VGG16
    images = extract_features(df)
    # split into training and testing data
    prop_test, prop_val = 0.2, 0.2

    N = df.shape[0]
    Ntest, Nval = int(N * prop_test), int(N * prop_val)

    def split_test_val_train(data_list, Ntest, Nval):
        return (data_list[:Ntest],
                data_list[Ntest:Ntest + Nval],
                data_list[Ntest + Nval:])

    dt_test, dt_val, dt_train = split_test_val_train(df['Caption'], Ntest, Nval)
    di_test, di_val, di_train = split_test_val_train(images.values(), Ntest,
                                                     Nval)
    fnm_test, fnm_val, fnm_train = split_test_val_train(images.keys(), Ntest,
                                                        Nval)

    # Find the max length of the caption
    config.maxlen = np.max([len(text) for text in df['Caption']])
    print('Maximum length of caption:',config.maxlen)

    # maximum number of words in dictionary
    nb_words = 6000
    # Tokenize data
    tokenizer = Tokenizer(nb_words=nb_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['Caption'][Ntest:])
    config.vocab_size = len(tokenizer.word_index) + 1
    print('vocabulary size', config.vocab_size)
    dt_train = tokenizer.texts_to_sequences(dt_train)
    dt_test = tokenizer.texts_to_sequences(dt_test)
    dt_val = tokenizer.texts_to_sequences(dt_val)

    # Obtain image feature and captions divided in seq2seq format
    Xtext_train, Ximage_train, ytext_train = preprocessing(dt_train,
                                                           di_train)
    Xtext_val, Ximage_val, ytext_val = preprocessing(dt_val, di_val)
    # pre-processing is not necessary for testing data
    # Xtext_test,  Ximage_test,  ytext_test  = preprocessing(dt_test, di_test)

    # final neural network model
    img_text_model = img_to_text_model(Ximage_train.shape[1])

    # Train model
    start = time.time()
    hist = img_text_model.fit([Ximage_train, Xtext_train], ytext_train,
                              epochs=config.EPOCHS, verbose=2,
                              batch_size=config.BATCH_SIZE, validation_data=(
            [Ximage_val, Xtext_val], ytext_val))
    end = time.time()
    print('Time of execution = {:3.2f}MIN'.format((end - start) / 60))
    print('Dimensions of image, input_text and output_text:')
    print(Ximage_train.shape, Xtext_train.shape, ytext_train.shape)
    # Display metric graphs
    metric(hist)

    # Inverse of tokenizer.word_index dictionary
    index_word = dict([(index, word) for word, index in tokenizer.word_index])
    # plot images along with their captions
    plot_images(img_text_model, fnm_test, di_test, tokenizer, index_word)
    # Display BLEU score
    bleu_score(img_text_model, fnm_test, di_test, dt_test, tokenizer,
               index_word)


if __name__ == '__main__':
    print('Test')
    main()
