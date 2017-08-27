#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Open data files from pickles
with open('experiment_word_data.pkl', 'r') as word_file:
    word_data = pickle.load(word_file)

with open('poi_labels.pkl', 'r') as label_file:
    label_data = pickle.load(label_file)

# Split train and test data
features_train, features_test, labels_train, labels_test = train_test_split(
    word_data, label_data, test_size=0.3)

# Vectorize features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, max_features=200,
    stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)


# Fit and score classifier
clf = MultinomialNB()
clf.fit(features_train, labels_train)

print 'Score on train set: ' + str(clf.score(features_train, labels_train))
print 'Score on test set: ' + str(clf.score(features_test, labels_test))

