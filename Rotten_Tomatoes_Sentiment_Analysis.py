# Importing required packages
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from keras_preprocessing import text


# Methods
def remove_empty_and_null_data(df, col):
    """
    Removes empty and null data

    Parameters
    ----------
    df : TYPE
        Dataframe.
    col : TYPE
        string.

    Returns
    -------
    df : TYPE
        Dataframe.

    """
    df= df.dropna()
    df = df[df[col].apply(lambda x: x != "")]
    df = df[df[col].apply(lambda x: x != " ")]
    return df

def clean_data(text):
    """
    Cleans the text 
        1. By removing punctuation, numbers, symbols
        2. Converting words into its base words
        

    Parameters
    ----------
    text : TYPE
        string.

    Returns
    -------
    updated_text : TYPE
        string.

    """
    # Remove puncuation
    text = text.translate(string.punctuation)
    
    # Convert words to lower case and split them
    text = text.lower().split()
    
    updated_text = []
    
    for word in text:
        word = str(word)
        word = re.sub("[^A-Za-z0-9^,!.\/'+-=]", " ", word)
        word = re.sub("what's", "what is ", word)
        word = re.sub("\'s", " ", word)
        word = re.sub("\'ve", " have ", word)
        word = re.sub("n't", " not ", word)
        word = re.sub("i'm", "i am ", word)
        word = re.sub("\'re", " are ", word)
        word = re.sub("\'d", " would ", word)
        word = re.sub("\'ll", " will ", word)
        word = re.sub(",", " ", word)
        word = re.sub("\.", " ", word)
        word = re.sub("!", " ! ", word)
        word = re.sub("\/", " ", word)
        word = re.sub("\^", " ^ ", word)
        word = re.sub("\+", " + ", word)
        word = re.sub("\-", " - ", word)
        word = re.sub("\=", " = ", word)
        word = re.sub("'", " ", word)
        word = re.sub("(\d+)(k)", r"\g<1>000", word)
        word = re.sub(":", " : ", word)
        word = re.sub(" e g ", " eg ", word)
        word = re.sub(" b g ", " bg ", word)
        word = re.sub(" u s ", " american ", word)
        word = re.sub("\0s", "0", word)
        word = re.sub(" 9 11 ", "911", word)
        word = re.sub("e - mail", "email", word)
        word = re.sub("j k", "jk", word)
        word = re.sub("\s{2,}", " ", word)
        
        if word != ' ' and word != '':
            updated_text.append(word)
        
    # Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lem_words = [wordnet_lemmatizer.lemmatize(w) for w in updated_text]
    updated_text = " ".join(lem_words)
    
    return updated_text
            

# Loading Data
train_data = pd.read_csv('train.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')

full_text = list(train_data['Phrase'].values) + list(test_data['Phrase'].values)

# Data Preprocessing
train_data = remove_empty_and_null_data(train_data, 'Phrase')
train_data['Phrase'] = train_data['Phrase'].map(lambda x: clean_data(x))
train_data = remove_empty_and_null_data(train_data, 'Phrase')
x = train_data['Phrase']

y = to_categorical(train_data['Sentiment'].values)

test_data = remove_empty_and_null_data(test_data, 'Phrase')
test_data['Phrase'] = test_data['Phrase'].map(lambda x: clean_data(x))
test_data = remove_empty_and_null_data(test_data, 'Phrase')
test_x = test_data['Phrase']


# Tokenizing & Padding
tokenizer = Tokenizer(lower = True, num_words=20000)
tokenizer.fit_on_texts(x)

x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=125)

test_x = tokenizer.texts_to_sequences(test_x)
test_x = pad_sequences(test_x, maxlen=125)

train_x, x_val, train_y, y_val = train_test_split(x, y, test_size=0.2, random_state=123,
                                                  stratify=y)

# Building Model
model = Sequential()
model.add(Embedding(20000,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()

model.fit(train_x, train_y, validation_data=(x_val, y_val), batch_size=1000, epochs=8, verbose=1)

# Predection
predection_file = test_data.copy()
predection_file['Sentiment'] = model.predict_classes(test_x, batch_size=1000, verbose=1)
predection_file.to_csv('SentimentPredection.csv', index=False)