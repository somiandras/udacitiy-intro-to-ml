import pandas as pd
import numpy as np

payments_features = ['salary',
                    'bonus',
                    'long_term_incentive',
                    'deferred_income',
                    'deferral_payments',
                    'loan_advances',
                    'other',
                    'expenses',
                    'director_fees']


def scoring(row):
    score = 0
    for feature in payments_features:
        score += ~np.isnan(row[feature])
    return score


def add_new_features(data):
    df = pd.DataFrame(data)
    df = df.transpose()
    df.replace(to_replace='NaN', value=np.nan, inplace=True)

    # Financial ratios
    df['adjusted_payments'] = df['total_payments'] - df['deferral_payments']
    df['payment_to_stock_value_ratio'] = df['total_stock_value'] /\
        df['adjusted_payments']
    df['salary_ratio'] = df['salary'] / df['adjusted_payments']
    df['payments_score'] = df.apply(scoring, axis=1)

    for feature in payments_features:
        df[feature].replace(np.nan, value=0, inplace=True)

    # Email ratios
    df['outbox_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
    df['inbox_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
    df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
    df['poi_inbox_outbox_ratio'] = df['from_poi_to_this_person'] /\
        df['from_this_person_to_poi']

    # Clean-up, return dictionary
    df.replace(np.inf, value=np.nan, inplace=True)
    df.replace(to_replace=np.nan, value='NaN', inplace=True)
    return df.transpose().to_dict()
