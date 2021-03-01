'''
Configuration file: contains general parameters
'''
import os

BATCH_SIZE = 32
EPOCHS = 5
npix = 224
target_size = (npix, npix, 3)
vocab_size = None
maxlen = None

BASE_INPUT = 'data/instagram_data'
BASE_OUTPUT = 'output'

# use the base output path to derive the path to the serialized model along
# with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'Image_to_caption'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
