import os
import numpy as np
import pickle

from Classifier import createModel, getFilteredText
from TextExtraction import extractTextForPrediction
from Classifier import tokenizeSentence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def pred(filepath):

    model = createModel()
    model.load_weights("../model/model")

    text = getFilteredText(extractTextForPrediction(filepath))

    with open('../model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=250)

    pred = model.predict(padded)

    labels = [x[1] for x in os.walk("../data")][0]
    print(pred, labels[np.argmax(pred)])
    print(" ")

if __name__== "__main__":
    # pred('C:\Git\MSE\Classifier\\temp\pred\Physiotherapie (deutsch).pdf44.pdf')
    # pred('C:\Git\MSE\Classifier\\temp\pred\MiGel (deutsch).pdf0.pdf')
    pred('C:\Git\MSE\Classifier\\temp\pred\Pflegefachpersonen (deutsch).pdf2.pdf')
