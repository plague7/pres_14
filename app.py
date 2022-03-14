import os
import os.path

import streamlit as st
from tensorflow import keras
from tensorflow.keras.utils import *
import tensorflow as tf
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAvgPool1D

#LOADING MODEL#
#Loading the model#

#Test#
#model = load_model("C:/Users/Simplon/Desktop/Travaux python/Présentation 14 Mars/matthis_model.h5")
#When deploying#
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'matthis_model.h5')
model = keras.models.load_model(MODEL_DIR)

#STOPWORDS#
stop_words = set(stopwords.words('english'))
stop_words.remove('not')

#TOKENIZER#
#with open("C:/Users/Simplon/Desktop/Travaux python/Présentation 14 Mars/tokenizer.pickle" , 'rb') as handle:
#    tokenizer = pickle.load(handle)
TOKENIZER_DIR = os.path.join(os.path.dirname('__file__'), 'tokenizer.pickle')
with open(TOKENIZER_DIR , 'rb') as handle:
    tokenizer = pickle.load(handle)

#Some random shit#
maxlen = int(50)
#Padded_train = pad_sequences(Tokenized_train, maxlen=maxlen, padding='pre')
#Padded_val = pad_sequences(Tokenized_val, maxlen=maxlen, padding='pre')

#FUNCTIONS#

#Remove punctuations#
def remove_punctuations_numbers(inputs):
    return re.sub(r'[^a-zA-Z]', ' ', inputs)

#Tokenization of the input#
def tokenization(inputs):  # Ref.1
    return word_tokenize(inputs)

#Removing stopwords#
def stopwords_remove(inputs):  # Ref.2
    return [k for k in inputs if k not in stop_words]

#Lemmatizer#
def lemmatization(inputs):  # Ref.1
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]

#Removing small words (less than 2)#
def remove_less_than_2(inputs):  # Ref.1
    inputs = [j for j in inputs if len(j) > 2]

#Predict the written review#
def predict_recommendation(input_text):  # The function for doing all the previous steps
    input_text = input_text.lower()
    input_text = re.sub(r'[^a-zA-Z]', ' ', input_text)
    input_text = tokenization(input_text)
    input_text = stopwords_remove(input_text)
    input_text = lemmatization(input_text)
    input_text = ' '.join(input_text)
    input_text = tokenizer.texts_to_sequences([input_text])
    input_text = pad_sequences(input_text, maxlen=maxlen, padding='pre')
    prediction = model.predict(input_text)
    if prediction >= 0.5:
        st.write('Positive review')
    else:
        st.write('Negative review')


#STREAMLIT#

#STREAMLIT FUNCTIONS#

def user_input():
    st.write('Write a common review you could find on an online woman clothing shop.')
    st.write('Then, the app will try to detect if you are giving it a good review, or not')
    text = str(st.text_input('Type your review here !'))
    return text

#Header#
st.title('Welcome to our online sentiment analysis app!')

### Selectbox ###
### Randomizing Tool View ###
text=user_input()
if st.button('Check the review'):
    #input_analysis(text)
    predict_recommendation(text)
    