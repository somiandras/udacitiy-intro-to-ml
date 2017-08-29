#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from feature_creator import add_new_features
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score,\
    confusion_matrix

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
        'poi',
        'logPayments',
        'total_stock_value',
        'total_benefits',
        'bonus',
        'salary',
        'exercised_stock_options',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'shared_receipt_with_poi',
        'shared_poi_ratio',
        'poi_inbox_outbox_ratio'
        ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

# Pop 'TOTAL' from the data
data_dict.pop('TOTAL', None)

### Task 3: Create new feature(s)

data_dict = add_new_features(data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Create pipeline
scaler = MinMaxScaler()
pca = PCA(random_state=42)
tree = DecisionTreeClassifier(criterion='entropy', random_state=42)

pipeline = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('tree', tree)
])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Parameter grid for search
parameters = {
    'pca__n_components': list(range(1, len(features_list))),
    'tree__min_samples_split': list(range(2, 20))
}

# 'Brute force' search for best n_components and min_samples_split values
grid_search = GridSearchCV(pipeline,
    param_grid=parameters,
    scoring='recall')
grid_search.fit(features, labels)

print 'Best estimator:'
print grid_search.best_estimator_

# Keep the best modell as classifier
clf = grid_search.best_estimator_

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Fit the classifier to the train dataset
clf.fit(features_train, labels_train)
# Predict labels on the test dataset
prediction = clf.predict(features_test)

# Print basic metrics on test set
print 'Precision: {0}, Recall: {1}, Accuracy: {2}'.format(
    precision_score(prediction, labels_test),
    recall_score(prediction, labels_test),
    accuracy_score(prediction, labels_test)
    )

# Print the confusion matrix
print('Confusion matrix:')
print confusion_matrix(prediction, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
