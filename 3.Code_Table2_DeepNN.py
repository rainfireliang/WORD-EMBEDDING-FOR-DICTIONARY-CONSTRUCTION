# -*- coding: utf-8 -*-
"""
This is the code to replciate Table 2 (deep learning models)

"""
from sklearn import model_selection, preprocessing, metrics
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

## load traning datasets (manually coded)
df = pd.read_csv('train.csv',encoding="utf-8")
df = df.dropna()
df['incivility'] = np.where(df['incivility']>0,1,0) # manually coded labels >1:uncivil; 0: civil

# load the pre-trained word-embedding vectors 
w2v_model = Word2Vec.load("word2vec_hk_2022.model")

## load Cantonese stop words
stop_words = [x.strip() for x in open('stopCantonese.txt',encoding="utf8").read().split('\n')]
## load the created dictionary: uncivil_words(N=1,956)
uncivil_words = [x.strip() for x in open('uncivil_words.txt',encoding="utf8").read().split('\n')]
## DIC: whether it is uncivil accroding to the dictioanry
df['DIC'] = df['txt_cleaned'].map(lambda j:int(any(ele in j for ele in uncivil_words)))

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(df['doc'])

# create token-embedding mapping
embedding_matrix = np.zeros((len(token.word_index) + 1, 250))
for word, i in token.word_index.items():
        try:
            embedding_vector = w2v_model.wv[word]
        except:
            embedding_vector = None
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

##########Model Building#############
        
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)    
    predictions = [1 if p>0.5 else 0 for p in predictions]    
    return metrics.accuracy_score(predictions, valid_y), metrics.precision_score(predictions, valid_y), metrics.recall_score(predictions, valid_y), metrics.f1_score(predictions, valid_y)


def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((500, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(token.word_index) + 1, 250, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(250, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy') #optimizers.Adam()
    
    return model

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((500, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(token.word_index) + 1, 250, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(250)(embedding_layer)

    # Add the output Layers

    output_layer1 = layers.Dense(64, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy')
    
    return model

def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((500, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(token.word_index) + 1, 250, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(128)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model

def create_rcnn():
    # Add an Input Layer
    input_layer = layers.Input((500, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(token.word_index) + 1, 250, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)
    
    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)
    
    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(250, 3, activation="relu")(rnn_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer="adam", loss='binary_crossentropy')
    
    return model


##############
def cross_validation(df,classifier):
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['doc'].values, df['incivility'].values, test_size=0.33) 
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    # convert text to sequence of tokens and pad them to ensure equal length vectors 
    train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=500)
    valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=500)
    
    accuracy,precision,recall,F = train_model(classifier, train_seq_x, train_y, valid_seq_x, valid_y)
    return accuracy,precision,recall,F

cross_validation(df, create_cnn())
cross_validation(df, create_rnn_lstm())
cross_validation(df, create_rnn_gru())
cross_validation(df, create_rcnn())

#
CNN_accuracy = []
CNN_precision = []
CNN_recall = []
CNN_F = []

#
LSTM_accuracy = []
LSTM_precision = []
LSTM_recall = []
LSTM_F = []

#
GRU_accuracy = []
GRU_precision = []
GRU_recall = []
GRU_F = []

#
RCNN_accuracy = []
RCNN_precision = []
RCNN_recall = []
RCNN_F = []

for i in range(10):
    accuracy,precision,recall,F = cross_validation(df, create_cnn())
    CNN_accuracy.append(accuracy)
    CNN_precision.append(precision)
    CNN_recall.append(recall)
    CNN_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, create_rnn_lstm())
    LSTM_accuracy.append(accuracy)
    LSTM_precision.append(precision)
    LSTM_recall.append(recall)
    LSTM_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, create_rnn_gru())
    GRU_accuracy.append(accuracy)
    GRU_precision.append(precision)
    GRU_recall.append(recall)
    GRU_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, create_rcnn())
    RCNN_accuracy.append(accuracy)
    RCNN_precision.append(precision)
    RCNN_recall.append(recall)
    RCNN_F.append(F)

sum(CNN_accuracy)/10 # 0.7308324768756423
sum(CNN_F)/10 # 0.779627020526408
sum(CNN_precision)/10 # 0.832255701414509
sum(CNN_recall)/10 # 0.7454848279215163

sum(LSTM_accuracy)/10 # 0.7772867420349434
sum(LSTM_F)/10 # 0.8132527498892538
sum(LSTM_precision)/10 # 0.8377014282004328
sum(LSTM_recall)/10 # 0.791579951158774

sum(GRU_accuracy)/10 # 0.6718396711202466
sum(GRU_F)/10 # 0.7345248199631167
sum(GRU_precision)/10 # 0.7852549697084118
sum(GRU_recall)/10 # 0.6912483501224714

sum(RCNN_accuracy)/10 # 0.7776978417266187
sum(RCNN_F)/10 # 0.8246696430256117
sum(RCNN_precision)/10 # 0.894932299233707
sum(RCNN_recall)/10 # 0.7662524964216939

