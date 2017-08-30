#!/usr/bin/python

import pandas as pd
import numpy as np

financial_features = [
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
    'total_stock_value'
]

email_features = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages'
]

payment_features = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'salary'
]


def scoring(row):
    '''
    Count how many of the payment features exists for the person
    '''

    score = 0
    for feature in payment_features:
        score += row[feature] != 0
    return score


def clean_data(data):
    '''
    Correct known data errors in initial data dictionary
    '''

    entries_to_remove = [
                    'TOTAL', 
                    'LAY KENNETH L',
                    'SHAPIRO RICHARD S',
                    'KAMINSKI WINCENTY J',
                    'GLISAN JR BEN F',
                    'KEAN STEVEN J' ]

    for entry in entries_to_remove:
        data.pop(entry, None)

    # Correct two entries based on the pdf report
    data['BHATNAGAR SANJAY'] = {'bonus': 'NaN',
     'deferral_payments': 'NaN',
     'deferred_income': 'NaN',
     'director_fees': 'NaN',
     'email_address': 'sanjay.bhatnagar@enron.com',
     'exercised_stock_options': 15456290,
     'expenses': 137864,
     'from_messages': 29,
     'from_poi_to_this_person': 0,
     'from_this_person_to_poi': 1,
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 'NaN',
     'poi': False,
     'restricted_stock': -2604490,
     'restricted_stock_deferred': 2604490,
     'salary': 'NaN',
     'shared_receipt_with_poi': 463,
     'to_messages': 523,
     'total_payments': 137864,
     'total_stock_value': 15456290}

    data['BELFER ROBERT'] = {'bonus': 'NaN',
     'deferral_payments': -102500,
     'deferred_income': 'NaN',
     'director_fees': 102500,
     'email_address': 'NaN',
     'exercised_stock_options': 'NaN',
     'expenses': 3285,
     'from_messages': 'NaN',
     'from_poi_to_this_person': 'NaN',
     'from_this_person_to_poi': 'NaN',
     'loan_advances': 'NaN',
     'long_term_incentive': 'NaN',
     'other': 'NaN',
     'poi': False,
     'restricted_stock': 44093,
     'restricted_stock_deferred': -44093,
     'salary': 'NaN',
     'shared_receipt_with_poi': 'NaN',
     'to_messages': 'NaN',
     'total_payments': 3285,
     'total_stock_value': 'NaN'}

    return data


def add_new_features(data):
    '''
    Add new features to each person in the data_dict.
    The data_dict is converted to pandas DataFrame for easier
    calculations, and then converted back to dictionary again.
    '''

    df = pd.DataFrame(data)
    df = df.transpose()
    df.replace(to_replace='NaN', value=np.nan, inplace=True)

    # Replace missing values with 0 in financial features
    df[financial_features] = df[financial_features].fillna(0)

    # Add flags for financial and email outliers
    df['email_outlier'] = ((df['from_messages'] > 10000) |
                           (df['to_messages'] > 10000) |
                           (df['to_messages'] < df['shared_receipt_with_poi']))
    df['financial_outlier'] = (df['total_payments'] > 100000000)

    # Financial ratios
    df['adjusted_payments'] = df['total_payments'] - df['deferred_income']
    df['payment_to_stock_value_ratio'] = df['total_stock_value'] /\
        df['adjusted_payments']
    df['payments_score'] = df.apply(scoring, axis=1)
    df['total_benefits'] = df['total_stock_value'] + df['total_payments']

    # Email ratios
    df['total_email_traffic'] = df['from_messages'] + df['to_messages']
    df['poi_email_traffic'] = df['from_this_person_to_poi'] + \
                                df['from_poi_to_this_person']
    df['outbox_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
    df['inbox_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']


    # Clean-up, return dictionary
    df.replace(np.inf, value=np.nan, inplace=True)
    df.replace(to_replace=np.nan, value='NaN', inplace=True)
    return df.transpose().to_dict()
