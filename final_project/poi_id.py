#!/usr/bin/python
import pprint as pp
import pickle
import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from process_data import clean_data, add_new_features

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, accuracy_score,\
    confusion_matrix

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
        'poi',
        'from_messages',
        'from_poi_to_this_person',
        'from_this_person_to_poi',
        'shared_receipt_with_poi',
        'to_messages',
        'bonus',
        'deferral_payments',
        'deferred_income',
        'director_fees',
        'exercised_stock_options',
        'expenses',
        'loan_advances',
        'long_term_incentive',
        'other',
        'restricted_stock',
        'restricted_stock_deferred',
        'salary',
        'total_payments',
        'total_stock_value',

        # Engineered features
        'total_email_traffic',
        'poi_email_traffic',
        'outbox_poi_ratio',
        'inbox_poi_ratio',
        'adjusted_payments',
        'payment_to_stock_value_ratio',
        'payments_score',
        'total_benefits'
        ]

final_features_list = [
        'poi',
        'shared_receipt_with_poi',
        'bonus',
        'deferred_income',
        'exercised_stock_options',
        'salary',
        'total_stock_value',
        'outbox_poi_ratio',
        'payments_score'
        ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

cleaned_data = clean_data(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = add_new_features(cleaned_data)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Create pipeline with SelectKBest
select_pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('selector', SelectKBest(k='all'))
])

# Extract the scores for the features in selector
select_pipeline.fit(features, labels)
fitted_selector = select_pipeline.get_params()['selector']

# Pretty print the scores
print '\nSCORES FROM SELECTKBEST:\n'
pp.pprint(sorted(
    zip(features_list[1:],
    fitted_selector.scores_,
    fitted_selector.pvalues_
    ), key=lambda x: x[1], reverse=True))


# Try different models in a similar pipeline with k=8 features
models = {
    'dt':  DecisionTreeClassifier(),
    'nb': GaussianNB(),
    'svc': SVC()
}

results = []
for model in models:
    pipe = Pipeline([
        ('scaler', MinMaxScaler()),
        ('selector', SelectKBest(k=8)),
        ('classifier', models[model])
    ])

    pipe.fit(features, labels)

    results.append((model,
        precision_score(pipe.predict(features), labels),
        recall_score(pipe.predict(features), labels)))

print '\nAVERAGE RESULTS FROM BASIC MODELS WITH K=8:'
print results

### Extract final features and labels from dataset
final_data = featureFormat(my_dataset, final_features_list, sort_keys=True)
final_labels, final_features = targetFeatureSplit(final_data)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print '\nBuilding models for k=1,2,...8 values. This takes a while...'

estimators = []
# Try 2-8 best features with PCA and DT classifier
for k_value in range(2, 9):
    # Create pipeline with PCA
    final_pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('k_best', SelectKBest(k=k_value)),
        ('selector', PCA()),
        ('tree', DecisionTreeClassifier())
    ])

    # Parameter grid for DT param search
    tree_parameters = {
        'selector__n_components': list(range(2, k_value + 1)),
        'tree__min_samples_split': list(range(2, 30)),
        'tree__min_samples_leaf': list(range(2, 10))
    }

    # Search for best features set and min_samples_split value
    # with Decision Tree
    grid_search_tree = GridSearchCV(final_pipeline,
        param_grid=tree_parameters,
        scoring='recall')

    grid_search_tree.fit(final_features, final_labels)

    # Store the best estimator for the given k value
    estimators.append((
        k_value,
        grid_search_tree.best_score_,
        grid_search_tree.best_params_, 
        grid_search_tree.best_estimator_))

# Get the estimators sorted by recall score
estimators.sort(key=lambda x: x[1], reverse=True)
for k, score, params, estimator in estimators:

    print '\nBEST PARAMETERS FOR {} BEST FEATURES:'.format(k)
    print 'Score: {}'.format(score)
    print 'Parameters: {}'.format(params)
    print '---------------'

# Keep the best modell as classifier
clf = estimators[0][3]

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

print '\nSTRATIFIED SHUFFLE SPLIT CV:'
kf = StratifiedShuffleSplit(n_splits=4, test_size=0.2)
results = []
for train_index, test_index in kf.split(final_features, final_labels):
    features_train = [final_features[idx] for idx in train_index]
    features_test = [final_features[idx] for idx in test_index]
    labels_train = [final_labels[idx] for idx in train_index]
    labels_test = [final_labels[idx] for idx in test_index]

    # Fit the classifier to the train dataset
    clf.fit(features_train, labels_train)

    # Predict labels on the test dataset
    prediction = clf.predict(features_test)

    # Print the confusion matrix
    print('Confusion matrix:')
    print confusion_matrix(prediction, labels_test)

    # Store scores
    results.append((
        precision_score(prediction, labels_test),
        recall_score(prediction, labels_test),
        accuracy_score(prediction, labels_test)
    ))

# Print average metrics
prec, rec, acc = tuple(sum(score) / len(score) for score in zip(*results))
print '''\nRESULTS FOR STRATIFIED SHUFFLE SPLIT CV:
Average precision: {0}
Average recall: {1}
Average accuracy: {2}'''.format(prec, rec, acc)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, final_features_list)
