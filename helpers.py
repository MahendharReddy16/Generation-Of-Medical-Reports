import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sklearn
import tqdm
import nltk
import cv2
from sklearn.model_selection import train_test_split
import PIL
from PIL import Image
import time

import tensorflow as tf
import keras
from keras.layers import Input,concatenate,Dropout,LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras import Model
from tensorflow.keras import activations
import warnings
warnings.filterwarnings("ignore")
import nltk.translate.bleu_score as bleu
from tensorflow.keras.models import load_model

#loading the pretrained chexnet model for getting image features
final_chexnet_model=load_model("chexnet_final.h5")
#loading embedded matrix
embedding_matrix=np.load('Embedding_matrix.npy')
print(embedding_matrix.shape)
#Loading the pretrained tokenizer
import pickle
with open('tokenizer_1.pkl', 'rb') as handle:
    token= pickle.load(handle)
#loading the dict file with image names and corresponding reports to calculate bleu scores
with open('data_dict.pkl', 'rb') as handle:
    Data= pickle.load(handle)