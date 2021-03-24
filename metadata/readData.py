'''
Read and clean input data
'''

# import libraries
import pandas as pd
import numpy as np
from emoji import UNICODE_EMOJI
import re
import copy
import os
import string
from sklearn.utils import shuffle

def read_data(filename):
    '''
    Read dataframe
    :param filename: file name and path of data file
    :return: dataframe
    '''
    df = pd.read_csv(filename, sep=',', encoding='utf-8', header=0, index_col=0)
    # print('Data before cleaning')
    # print(df.head())
    df.dropna(axis=0, how='any', inplace=True)
    # print('/nData after cleaning')
    # print(df.head())
    return df

def isEnglishOrEmoji(s):
    '''
    Detect if English character or Emoji
    :param s: text
    :return: True if English character, False if Emoji
    '''
    try:
        for letter in s:
            if letter in UNICODE_EMOJI:
                s = s.replace(letter, "")
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def separateHashtags(s):
    '''
    Create list of Hashtags
    :param s:
    :return:
    '''
    hashtags_list = []
    s = re.split(' |\n|[\r\n]+',s)
    for word in s:
        if '#' in word:
            hashtags_list.append(word)
    return hashtags_list

def removeHashtags(s):
    '''
    Remove all hashtags from text
    :param s: text
    :return: text without hashtag
    '''
    caption = []
    s = re.split(' |\n|[\r\n]+',s)
    for word in s:
        if '#' not in word:
            caption.append(word)
    return " ".join(caption)

def removeLinksAndTags(s):
    '''
    Remove links and hashtags
    :param s: text
    :return: text without link and hashtag
    '''
    caption = []
    s = re.split(' |\n|[\r\n]+',s)
    for word in s:
        if 'https:' not in word and '@' not in word and '.com' not in word:
            caption.append(word)
    return " ".join(caption)

def isEmoji(s):
    '''
    Detect if a character is emoji
    :param s: text
    :return: list of all the emojis collected
    '''
    emojis = []
    for letter in s:
        if letter in UNICODE_EMOJI:
            emojis.append(letter)
    return emojis

def add_tags(captions):
    '''
    Add start and end tags to every caption text
    :param captions: caption/text
    :return: caption with <START>, <END> tags
    '''
    caps = []
    for txt in captions:
        txt = '<START>' + txt + '<END>'
        caps.append(txt)
    return caps

def remove_single_character(text):
    '''
    Remove captions with just one character
    :param text: text/caption
    :return: blank if only one character is present
    '''
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

def remove_punctuation(text_original):
    '''
    Remove punctuation from captions
    :param text_original: captions/text
    :return: captions without punctuation
    '''
    # text_no_punctuation = text_original.translate(str.maketrans('','',string.punctuation))
    text_no_punctuation = re.sub(r'[^\w\s]', '', text_original)
    return(text_no_punctuation)

def load_images_list(directory):
    '''
    list of all images in the directory
    :param directory: directory path
    :return: list of all the images
    '''
    images = os.listdir(directory)
    return images

def clean_data():
    print('[INFO] reading data')
    df = read_data('data/instagram_data/captions_csv.csv')
    # df2 = read_data('data/instagram_data/captions_csv2.csv')
    # df = pd.concat([df, df2])
    df = shuffle(df)
    # print('\n\nNew merged Dataframe')
    # print(df.head())
    # print('column names:', df.columns)
    # print('Number of rows before emoji cleaning:', len(df))
    # df["english"] = df['Caption'].map(isEnglishOrEmoji)
    # df = df[df.english == True]
    df['Caption'] = df['Caption'].map(remove_punctuation)
    df['Caption'] = df['Caption'].map(remove_single_character)
    df['Caption'] = df['Caption'].map(removeLinksAndTags)
    df['Caption'] = df['Caption'].map(removeHashtags)

    # remove rows with blank cells
    df.dropna(axis=0, how='any', inplace=True)
    # print('Number of rows after emoji cleaning:', len(df))

    # add tags at start and end of each caption
    df['Caption'] = add_tags(df['Caption'])
    return df
