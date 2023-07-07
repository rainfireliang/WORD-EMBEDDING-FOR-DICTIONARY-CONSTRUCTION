# -*- coding: utf-8 -*-
"""
This is the code to replciate Table 2 (average of word2vec)

"""
from sklearn import model_selection, linear_model, naive_bayes, metrics, svm, preprocessing
from sklearn import ensemble
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import xgboost


## load traning datasets (manually coded)
df = pd.read_csv('train.csv',encoding="utf-8")
df = df.dropna()
df['incivility'] = np.where(df['incivility']>0,1,0) # manually coded labels >1:uncivil; 0: civil

# load the pre-trained word-embedding vectors 
w2v = Word2Vec.load("word2vec_hk_2022.model")

## load Cantonese stop words
stop_words = [x.strip() for x in open('stopCantonese.txt',encoding="utf8").read().split('\n')]
## load the created dictionary: uncivil_words(N=1,956)
uncivil_words = [x.strip() for x in open('uncivil_words.txt',encoding="utf8").read().split('\n')]
## DIC: whether it is uncivil accroding to the dictioanry
df['DIC'] = df['txt_cleaned'].map(lambda j:int(any(ele in j for ele in uncivil_words)))

## document vector  = average word2vec
def document_vector(doc):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = doc.split()
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word in w2v.wv.index_to_key]
    
    if len(doc)>0:
        x = np.mean(w2v.wv[doc], axis=0)
    else:
        x = np.zeros(250)
    return x

df['doc_vector'] = df["doc"].apply(document_vector)

x_count = np.stack(df['doc_vector'])
scaler = preprocessing.StandardScaler().fit(x_count)

## accuracy calculation:
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid) 
    return metrics.accuracy_score(predictions, valid_y), metrics.precision_score(predictions, valid_y), metrics.recall_score(predictions, valid_y), metrics.f1_score(predictions, valid_y)

## cross-validation:
def cross_validation(df,classifier):
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['doc_vector'].values, df['incivility'].values, test_size=0.33)
       
    # transform the training and validation data using count vectorizer object
    xtrain_count =  np.stack(train_x, axis=0)
    xvalid_count =  np.stack(valid_x, axis=0)
    
    # scale
    accuracy,precision,recall,F = train_model(classifier, scaler.transform(xtrain_count), train_y, scaler.transform(xvalid_count),valid_y)
    return accuracy,precision,recall,F

############## experiments #################
NB_accuracy = []
NB_precision = []
NB_recall = []
NB_F = []

#
LR_accuracy = []
LR_precision = []
LR_recall = []
LR_F = []

#
SVM_accuracy = []
SVM_precision = []
SVM_recall = []
SVM_F = []

#
RF_accuracy = []
RF_precision = []
RF_recall = []
RF_F = []

#
XG_accuracy = []
XG_precision = []
XG_recall = []
XG_F = []

for i in range(10):
    accuracy,precision,recall,F = cross_validation(df, naive_bayes.BernoulliNB())
    NB_accuracy.append(accuracy)
    NB_precision.append(precision)
    NB_recall.append(recall)
    NB_F.append(F)  
    
    accuracy,precision,recall,F = cross_validation(df, linear_model.LogisticRegression(C=1/0.1,max_iter=1000))
    LR_accuracy.append(accuracy)
    LR_precision.append(precision)
    LR_recall.append(recall)
    LR_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, svm.SVC(kernel = "linear",C=1))
    SVM_accuracy.append(accuracy)
    SVM_precision.append(precision)
    SVM_recall.append(recall)
    SVM_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, ensemble.RandomForestClassifier())
    RF_accuracy.append(accuracy)
    RF_precision.append(precision)
    RF_recall.append(recall)
    RF_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, xgboost.XGBClassifier())
    XG_accuracy.append(accuracy)
    XG_precision.append(precision)
    XG_recall.append(recall)
    XG_F.append(F)



sum(NB_accuracy)/10 # 0.6793422404933196
sum(NB_F)/10 # 0.7141270090875163
sum(NB_precision)/10 # 0.6903572917868098
sum(NB_recall)/10 # 0.7399849166344157


sum(LR_accuracy)/10 # 0.721993833504625
sum(LR_F)/10 # 0.7631912659378088
sum(LR_precision)/10 # 0.7741092573275032
sum(LR_recall)/10 # 0.752990040382732


sum(SVM_accuracy)/10 # 0.7123329907502569
sum(SVM_F)/10 # 0.7575023913893222
sum(SVM_precision)/10 # 0.7826445319977952
sum(SVM_recall)/10 # 0.7344989579035823


sum(RF_accuracy)/10 # 0.7874614594039056
sum(RF_F)/10 # 0.8205205994432768
sum(RF_precision)/10 # 0.8311141671249562
sum(RF_recall)/10 # 0.810562198710658

sum(XG_accuracy)/10 # 0.776464542651593
sum(XG_F)/10 # 0.8137850340447654
sum(XG_precision)/10 # 0.8406505989974589
sum(XG_recall)/10 # 0.7891276811741263
