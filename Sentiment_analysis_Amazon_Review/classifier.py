#!/usr/bin/env python
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentence_polarity
import random
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import KFold

SLpath='subjclueslen1-HLTEMNLP05.tff'

documents = [(sent, cat) for cat in sentence_polarity.categories()for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, count) in word_items]

print("document_features")
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
featuresets = [(document_features(d,word_features), c) for (d,c) in documents]
train_set, test_set = featuresets[1000:], featuresets[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
classifier2.train(train_set)
print("NaiveBayes")
print (nltk.classify.accuracy(classifier, test_set))
print ("svm")
print (nltk.classify.accuracy(classifier2, test_set))

print("cross-validation")
avg_acc=0
avg_acc1=0
new=KFold(n_splits=5)
for train_index,test_index in new.split(featuresets):
    new_train,new_test=np.array(featuresets)[train_index],np.array(featuresets)[test_index]
    sl_classifier = nltk.NaiveBayesClassifier.train(new_train)
    sl_classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
    sl_classifier2.train(new_train)
    #print("svm")
    #print (nltk.classify.accuracy(classifier, new_test))
    #print ("NaiveBayes")
    #print (nltk.classify.accuracy(classifier2, new_test))
    avg_acc=avg_acc+nltk.classify.accuracy(classifier, new_test)
    avg_acc1=avg_acc1+nltk.classify.accuracy(classifier2, new_test)
print("NaiveBayes")
print(avg_acc/5)
print("svm")
print(avg_acc1/5)

print("Not_feature")
negationwords = ['no','not','never','none','nowhere','nothing','noone','rather','hardly','scarcely','rarely','seldom','neither','nor']
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features
NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in documents]
train_set, test_set = NOT_featuresets[1000:], NOT_featuresets[:1000]
not_classifier = nltk.NaiveBayesClassifier.train(train_set)
not_classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
not_classifier2.train(train_set)
print("not_NaiveBayes")
print (nltk.classify.accuracy(not_classifier, test_set))
print ("not_svm")
print (nltk.classify.accuracy(not_classifier2, test_set))

print("cross-validation")
avg_acc2=0
avg_acc3=0
nonew=KFold(n_splits=5)
for train_index,test_index in nonew.split(NOT_featuresets):
    nonew_train,nonew_test=np.array(NOT_featuresets)[train_index],np.array(NOT_featuresets)[test_index]
    not_classifier = nltk.NaiveBayesClassifier.train(nonew_train)
    not_classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
    not_classifier2.train(nonew_train)
    avg_acc2=avg_acc+nltk.classify.accuracy(not_classifier, nonew_test)
    avg_acc3=avg_acc3+nltk.classify.accuracy(not_classifier2, nonew_test)
print("NaiveBayes")
print(avg_acc2/5)
print("svm")
print(avg_acc3/5)

def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict
SL = readSubjectivity(SLpath)

print("SL_feature")
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
        return features

SL_featuresets = [(SL_features(d, word_features, SL), c) for (d,c) in documents]
train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
sl_classifier = nltk.NaiveBayesClassifier.train(train_set)
sl_classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
sl_classifier2.train(train_set)
print("sl_NaiveBayes")
print (nltk.classify.accuracy(sl_classifier, test_set))
print ("sl_svm")
print (nltk.classify.accuracy(sl_classifier2, test_set))

print("cross-validation")
avg_acc4=0
avg_acc5=0
slnew=KFold(n_splits=5)
for train_index,test_index in slnew.split(SL_featuresets):
    slnew_train,slnew_test=np.array(SL_featuresets)[train_index],np.array(SL_featuresets)[test_index]
    sl_classifier = nltk.NaiveBayesClassifier.train(slnew_train)
    sl_classifier2 = nltk.classify.SklearnClassifier(LinearSVC())
    sl_classifier2.train(slnew_train)
    avg_acc4=avg_acc4+nltk.classify.accuracy(classifier, slnew_test)
    avg_acc5=avg_acc5+nltk.classify.accuracy(classifier2, slnew_test)
print("NaiveBayes")
print(avg_acc4/5)
print("svm")
print(avg_acc5/5)



