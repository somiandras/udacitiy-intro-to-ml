#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score


# Open data files from pickles
with open('experiment_word_data.pkl', 'r') as word_file:
    word_data = pickle.load(word_file)

with open('poi_labels.pkl', 'r') as label_file:
    label_data = pickle.load(label_file)

# Split train and test data
features_train, features_test, labels_train, labels_test = train_test_split(
    word_data, label_data, test_size=0.5)

# Vectorize features
vectorizer = TfidfVectorizer(max_df=0.1, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)

# Fit and score classifier
clf = MultinomialNB()
clf.fit(features_train, labels_train)

print 'Score on train set: ' + str(clf.score(features_train, labels_train))
print 'Score on test set: ' + str(clf.score(features_test, labels_test))

labels_pred = clf.predict(features_test)
print confusion_matrix(labels_test, labels_pred)
print 'Precision: ' + str(precision_score(labels_test, labels_pred))
print 'Recall: ' + str(recall_score(labels_test, labels_pred))
print labels_test
print labels_pred
