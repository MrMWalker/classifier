import pandas as pd
import os
import glob
import re
import pickle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from nltk.corpus import stopwords
from nltk import flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def createModel():
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=250))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(18, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def getFilteredText(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;%$£\\-:=\!?.\,¢_\\n\'\\\]')
    SHORT_WORDS = re.compile(r'\W*\b\w{1,3}\b')
    #BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = SHORT_WORDS.sub('', text)
    #text = BAD_SYMBOLS_RE.sub('', text)
    return text



def getDataFrame():
    classes = [x[1] for x in os.walk("../data")][0]
    data = []

    for idx,c in enumerate(classes):
        class_path = "../data/" + c + "/"
        os.chdir(class_path)
        for text_file in glob.glob("*.txt"):
            file = open(text_file, mode='r')
            text = ''
            for _, line in enumerate(file):
                text += line
            text = getFilteredText(text)
            data.append([text, c, idx])
        os.chdir("../../code")
    return data

def tokenizeSentence(input):
    sentences = [re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl=' ', string=x).strip().split(' ')
                 for x in input.split('\n')
                 if not x.endswith('writes:')]
    sentences = [x for x in sentences if x != ['']]
    flat_list = flatten(sentences)
    flat_list = [x for x in flat_list if x != '']
    stopwords_german = set(stopwords.words('german'))
    filtered_tokens = [w for w in flat_list if not w in stopwords_german]
    return filtered_tokens

if __name__== "__main__":
    df = pd.DataFrame(getDataFrame(), columns=['text','class_name', 'class_label'])

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['text'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    Y = pd.get_dummies(df['class_label']).values
    print('Shape of label tensor:', Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)

    model = createModel()

    epochs = 17
    batch_size = 32

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    model.save("../model/model.h5")

    with open('../model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show();
