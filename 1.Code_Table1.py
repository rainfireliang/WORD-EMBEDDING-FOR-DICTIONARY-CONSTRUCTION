# -*- coding: utf-8 -*-
"""
This is the code to replciate Table 1

"""
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
import xgboost
import pandas as pd
import numpy as np


## load traning datasets (manually coded)
df = pd.read_csv('train.csv',encoding="utf-8")
df = df.dropna()
df['incivility'] = np.where(df['incivility']>0,1,0) # manually coded labels >1:uncivil; 0: civil

## load Cantonese stop words
stop_words = [x.strip() for x in open('stopCantonese.txt',encoding="utf8").read().split('\n')]
## load the created dictionary: uncivil_words(N=1,956)
uncivil_words = [x.strip() for x in open('uncivil_words.txt',encoding="utf8").read().split('\n')]
## DIC: whether it is uncivil accroding to the dictioanry
df['DIC'] = df['txt_cleaned'].map(lambda j:int(any(ele in j for ele in uncivil_words)))


## accuracy calculation:
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid) 
    return metrics.accuracy_score(predictions, valid_y), metrics.precision_score(predictions, valid_y), metrics.recall_score(predictions, valid_y), metrics.f1_score(predictions, valid_y)

## cross-validation:
def cross_validation(df,classifier,tfidf = False,xg = False):
    # split the dataset into training and validation datasets 
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['doc'].values, df['incivility'].values, test_size=0.33)
    
    # label encode the target variable 
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    
    # create a count vectorizer object 
    count_vect = CountVectorizer(tokenizer=lambda x: x.split(),stop_words= stop_words,min_df=2) #tokenizer=lambda x: x.split()
    count_vect.fit(df['doc']) #print(count_vect.get_feature_names())
    
    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(tokenizer=lambda x: x.split(),stop_words= stop_words,min_df=2)
    tfidf_vect.fit(df['doc'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)

    if xg:
        if tfidf:
            accuracy,precision,recall,F = train_model(classifier, xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc(),valid_y)
        else:
            accuracy,precision,recall,F = train_model(classifier, xtrain_count.tocsc(), train_y, xvalid_count.tocsc(),valid_y)
    else:
        if tfidf:
            accuracy,precision,recall,F = train_model(classifier, xtrain_tfidf, train_y, xvalid_tfidf,valid_y)
        else:
            accuracy,precision,recall,F = train_model(classifier, xtrain_count, train_y, xvalid_count,valid_y)
    return accuracy,precision,recall,F

############## experiments #################
NB_accuracy = []
NB_precision = []
NB_recall = []
NB_F = []

NB_accuracy_tfidf = []
NB_precision_tfidf = []
NB_recall_tfidf = []
NB_F_tfidf = []

#
LR_accuracy = []
LR_precision = []
LR_recall = []
LR_F = []

LR_accuracy_tfidf = []
LR_precision_tfidf = []
LR_recall_tfidf = []
LR_F_tfidf = []

#
SVM_accuracy = []
SVM_precision = []
SVM_recall = []
SVM_F = []

SVM_accuracy_tfidf = []
SVM_precision_tfidf = []
SVM_recall_tfidf = []
SVM_F_tfidf = []

#
RF_accuracy = []
RF_precision = []
RF_recall = []
RF_F = []

RF_accuracy_tfidf = []
RF_precision_tfidf = []
RF_recall_tfidf = []
RF_F_tfidf = []

#
XG_accuracy = []
XG_precision = []
XG_recall = []
XG_F = []

XG_accuracy_tfidf = []
XG_precision_tfidf = []
XG_recall_tfidf = []
XG_F_tfidf = []

for i in range(10):
    accuracy,precision,recall,F = cross_validation(df, naive_bayes.BernoulliNB(),tfidf=False)
    NB_accuracy.append(accuracy)
    NB_precision.append(precision)
    NB_recall.append(recall)
    NB_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, naive_bayes.BernoulliNB(),tfidf=True)
    NB_accuracy_tfidf.append(accuracy)
    NB_precision_tfidf.append(precision)
    NB_recall_tfidf.append(recall)
    NB_F_tfidf.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, linear_model.LogisticRegression(C=1/0.1),tfidf=False)
    LR_accuracy.append(accuracy)
    LR_precision.append(precision)
    LR_recall.append(recall)
    LR_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, linear_model.LogisticRegression(C=1/0.1),tfidf=True)
    LR_accuracy_tfidf.append(accuracy)
    LR_precision_tfidf.append(precision)
    LR_recall_tfidf.append(recall)
    LR_F_tfidf.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, svm.SVC(kernel = "linear",C=1),tfidf=False)
    SVM_accuracy.append(accuracy)
    SVM_precision.append(precision)
    SVM_recall.append(recall)
    SVM_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, svm.SVC(kernel = "linear",C=1),tfidf=True)
    SVM_accuracy_tfidf.append(accuracy)
    SVM_precision_tfidf.append(precision)
    SVM_recall_tfidf.append(recall)
    SVM_F_tfidf.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, ensemble.RandomForestClassifier(),tfidf=False)
    RF_accuracy.append(accuracy)
    RF_precision.append(precision)
    RF_recall.append(recall)
    RF_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df,  ensemble.RandomForestClassifier(),tfidf=True)
    RF_accuracy_tfidf.append(accuracy)
    RF_precision_tfidf.append(precision)
    RF_recall_tfidf.append(recall)
    RF_F_tfidf.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, xgboost.XGBClassifier(),tfidf=False, xg=True)
    XG_accuracy.append(accuracy)
    XG_precision.append(precision)
    XG_recall.append(recall)
    XG_F.append(F)
    
    accuracy,precision,recall,F = cross_validation(df, xgboost.XGBClassifier(),tfidf=True, xg=True)
    XG_accuracy_tfidf.append(accuracy)
    XG_precision_tfidf.append(precision)
    XG_recall_tfidf.append(recall)
    XG_F_tfidf.append(F)


sum(NB_accuracy)/10 # 0.698869475847893
sum(NB_F)/10 # 0.7641265424257353
sum(NB_precision)/10 # 0.8422285839194021
sum(NB_recall)/10 # 0.7014781305500875

sum(NB_accuracy_tfidf)/10 # 0.6978417266187049
sum(NB_F_tfidf)/10 # 0.7636150255352028
sum(NB_precision_tfidf)/10 # 0.8487064938595387
sum(NB_recall_tfidf)/10 # 0.6960884512178133

sum(LR_accuracy)/10 # 0.7595066803699897
sum(LR_F)/10 # 0.7816596481699061
sum(LR_precision)/10 # 0.745537465472736
sum(LR_recall)/10 # 0.8221797626943607

sum(LR_accuracy_tfidf)/10 # 0.7409044193216854
sum(LR_F_tfidf)/10 # 0.7758085521570987
sum(LR_precision_tfidf)/10 # 0.7801157560592503
sum(LR_recall_tfidf)/10 # 0.77177515551481

sum(SVM_accuracy)/10 # 0.7650565262076053
sum(SVM_F)/10 # 0.7854302901427703
sum(SVM_precision)/10 # 0.7402823505917948
sum(SVM_recall)/10 # 0.8366699443347171

sum(SVM_accuracy_tfidf)/10 # 0.7409044193216856
sum(SVM_F_tfidf)/10 # 0.7711489198558862
sum(SVM_precision_tfidf)/10 # 0.7511237886790018
sum(SVM_recall_tfidf)/10 # 0.7926423223110984

sum(RF_accuracy)/10 # 0.7738951695786228
sum(RF_F)/10 # 0.7908951327329403
sum(RF_precision)/10 # 0.7404524912281022
sum(RF_recall)/10 # 0.8492747344615204

sum(RF_accuracy_tfidf)/10 # 0.7723535457348406
sum(RF_F_tfidf)/10 # 0.7848637952972014
sum(RF_precision_tfidf)/10 # 0.7189270545962059
sum(RF_recall_tfidf)/10 # 0.8651227057192254

sum(XG_accuracy)/10 # 0.763617677286742
sum(XG_F)/10 # 0.7664141368922079
sum(XG_precision)/10 # 0.6748962340530901
sum(XG_recall)/10 # 0.8870776745242208

sum(XG_accuracy_tfidf)/10 # 0.751901336073998
sum(XG_F_tfidf)/10 # 0.7555499569798363
sum(XG_precision_tfidf)/10 # 0.671970215676915
sum(XG_recall_tfidf)/10 # 0.8633620716947135

metrics.accuracy_score(df['incivility'], df['DIC']) #0.9246691550729556
metrics.f1_score(df['incivility'], df['DIC']) #0.9345904537418974
metrics.precision_score(df['incivility'], df['DIC']) #0.940130409010077
metrics.recall_score(df['incivility'], df['DIC']) #0.9291154071470415
