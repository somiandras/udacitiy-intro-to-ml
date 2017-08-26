#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import string
import re
import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from extract_emails import get_sent_emails, extract_email_text, stem_text


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

email_addresses = [(data_dict[name]['email_address'], name) for
    name in data_dict if data_dict[name]['email_address'] != 'NaN']

for email_address, name in email_addresses[:15]:
    raw_emails = get_sent_emails(email_address)
    processed_emails = ' '.join({extract_email_text(email, name) for
        email in raw_emails})

    print processed_emails
