#!/usr/bin/env python
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentence_polarity
import random
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import KFold

documents = [(sent, cat) for cat in sentence_polarity.categories()for sent in sentence_polarity.sents(categories=cat)]
random.shuffle(documents)
all_words_list = [word for (sent,cat) in documents for word in sent]
all_words = nltk.FreqDist(all_words_list)
word_items = all_words.most_common(2000)
word_features = [word for (word, count) in word_items]

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

nonew=KFold(n_splits=5)
for train_index,test_index in nonew.split(NOT_featuresets):
    nonew_train,nonew_test=np.array(NOT_featuresets)[train_index],np.array(NOT_featuresets)[test_index]
    not_classifier = nltk.NaiveBayesClassifier.train(nonew_train)

stopwords = nltk.corpus.stopwords.words('english')
fstop = open('Smart.English.stop','r',encoding='utf -8')
mystoptext = fstop.read()
fstop.close()
mystopwords = nltk.word_tokenize(mystoptext)

wnl=nltk.WordNetLemmatizer()

myraw = open('baby.txt').read()
output1 = open('pos.txt', 'w',encoding='utf-8')
output2 = open('neg.txt', 'w',encoding='utf-8')
preview = re.findall(r"reviewText[:](.+?)\n\n",myraw,re.S)
for i in preview:
    preview2= re.findall(r"(.+?)[,] 2010",i,re.S)
    if(len(preview2)>0):
        review=re.findall(r"(.+?)\noverall",preview2[0])
        if(len(review)>0):
            sents=nltk.sent_tokenize(review[0])
            for it in sents:
                words=nltk.word_tokenize(it)
                alphafiltered=[w.lower() for w in words if w.isalpha()]
                stopfiltered = [w for w in alphafiltered if w not in stopwords and len(w)>2]
                stopfiltered2 = [w for w in stopfiltered if w not in mystopwords]
                Lemma = [wnl.lemmatize(w) for w in stopfiltered2]
                #le=set(Lemma)
                if(len(Lemma)>0):
                    res=not_classifier.classify(NOT_features(Lemma,word_features,negationwords))
                    if(res=="pos"):
                        output1.write(it)
                    if(res=="neg"):
                        output2.write(it)
            
output1.close()
output2.close()

print("--------end-------")

