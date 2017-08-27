#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import string
import re
import pickle
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(level=logging.INFO)


def get_sent_emails(email_address):
    '''
    Collect every email sent from the given email address.

    Params:
        email_address (string)

    Returns: list of strings of raw emails
    '''

    emails = []

    # Create path to email list txt
    path = 'emails_by_address/from_' + email_address + '.txt'
    try:
        with open(path, 'r') as email_list:
            # Loop through list file
            for line in email_list:
                # Change path to the correct directory
                path_elems = line.strip().split('/')
                new_path = '../' + '/'.join(path_elems[1:])

                # Open email file and add the contents to the list
                with open(new_path, 'r') as f:
                    emails.append(f.read())

    # No list for the email address
    except IOError as e:
        logging.error(e)

    return emails


def stem_text(text):
    '''
    Remove punctuation, lowercase and stem text

    Params:
        text: (string) The string to process

    Returns: (string) Stemmed words concatenated to one string
    '''

    cleaned = string.translate(text.lower(), None, string.punctuation)
    splitted = cleaned.split(' ')
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    new_text = []
    for word in splitted:
        new_text.append(stemmer.stem(word))
    return ' '.join(new_text)


def extract_email_text(email, name=None):
    '''
    Remove header data and forwarded messages and return the content of
    the actual email lowercased and stemmed. It tries to remove the
    sender's name too, if it is given (cannot handle nicknames and
    abbreviations)

    Params:
        email: (string) complete raw text of an extracted email
        name: (string, optional) name of sender to remove signatures

    Returns: (string) stemmed words concatenated with one whitespace
    '''

    text_to_return = ''

    # Split header from content
    delimiter = 'X-FileName:\s.*\n'
    parts = re.split(delimiter, email)
    if len(parts) > 1:
        # Try to find forwarded messages and split them
        # from the actual email
        original_split = re.split(
            '-*\s?(Original\sMessage|Forwarded\sby|From:.*\n|To:.*\n)',
            parts[1])

        if len(original_split) > 1:
            text_to_return = original_split[0]
        else:
            text_to_return = parts[1]
    else:
        logging.error('Cannot split email: ' + email[:100] + '...')
        text_to_return = email

    stemmed = stem_text(text_to_return)
    if name:
        stemmed_name = stem_text(name.strip()).split(' ')
        pattern = '|'.join(stemmed_name)
        cleaned_from_name = re.sub(pattern, '', stemmed)
        return cleaned_from_name
    else:
        return stemmed


def main():
    '''
    Load basic dataset and extract all outgoing emails for each email
    address. The emails are stemmed and cleaned and dumped to a
    pickle file. Poi labels are dumped to another pickle file.

    Params: None

    Returns: None

    Saves 'experiment_word_data.pkl' and 'poi_labels.pkl'.
    '''

    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # List of (email_adress, name) tuples from data_dict
    # excluding names without email address
    email_addresses = [(data_dict[name]['email_address'], name) for name
                    in data_dict if data_dict[name]['email_address'] != 'NaN']

    word_data = []
    poi_data = []

    for email_address, name in email_addresses:
        raw_emails = get_sent_emails(email_address)

        # Concat set of cleaned emails into one string
        # Set is used to exclude duplicate messages
        processed_emails = ' '.join({extract_email_text(email, name).strip()
            for email in raw_emails})
        if len(processed_emails) > 0:
            word_data.append(processed_emails)
            poi_data.append(int(data_dict[name]['poi']))
            logging.info(name + ' is processed')
        else:
            logging.info(name + ' has no emails to process')

    pickle.dump(word_data, open('experiment_word_data.pkl', 'w'))
    pickle.dump(poi_data, open('poi_labels.pkl', 'w'))


if __name__ == '__main__':
    main()
