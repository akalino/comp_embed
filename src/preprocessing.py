import re
import string

import pandas as pd

from nltk.corpus import stopwords
from sklearn.utils import shuffle

RANDOM_SEED = 17
OUTPUT_DIR = '/Users/ak/PycharmProjects/comp_embed/data/processed/'


def remove_pii(_text):
    """ A function to strip the documents of the CFPB scrubbed PII.
    :param _text: An individual document.
    :return: The same document without the masking strings.
    """
    return re.sub(r'\s*X+', '', _text)


def remove_punctuation(_text, _split):
    """ Using the string translate functionality to strip bad characters.
    :param _text: An individual document.
    :param _split: Binary flag to split string to tokens.
    :return: A cleaned document.
    """
    _text = _text.translate(str.maketrans('', '', string.punctuation))
    if _split:
        _text = _text.split(' ')
    return _text


def remove_stopwords(_text, _stops):
    """ A function to remove stopwords.
    :param _text: An input document.
    :param _stops: A list of stopwords.
    :return: The original list with all stopwords removed.
    """
    l = [w for w in _text.split(' ') if w not in _stops]
    return ' '.join(l)


def all_lower_single_line(_str):
    """
    A function for converting all characters to lower.
    :param _str: Input string.
    :return: Lower case string.
    """
    _str = _str.replace('\n', '')
    return _str.lower()


def clean_complaint_documents(_df, _stops):
    """ A function to apply language cleanup to a data frame.
    :param _df: The input data frame.
    :return _df: The input data frame with cleaned query columns.
    """
    doc_count = len(_df)
    _df['doc_id'] = [i+1 for i in range(doc_count)]
    _df['document'] = _df['text'].apply(lambda x: remove_punctuation(x, False))
    _df['document'] = _df['document'].apply(lambda x: remove_pii(x))
    _df['document'] = _df['document'].apply(lambda x: all_lower_single_line(x))
    _df['cleaned_document'] = _df['document'].apply(lambda x: remove_stopwords(x, _stops))
    return _df


if __name__ == "__main__":
    complaints = pd.read_csv('/Users/ak/PycharmProjects/comp_embed/data/raw/case_study_data.csv')
    stopwords = set(stopwords.words('english'))
    cleaned_complaints = clean_complaint_documents(complaints, stopwords)
    distinct_products = list(set(cleaned_complaints['product_group'].tolist()))
    for prod in distinct_products:
        subset = cleaned_complaints[cleaned_complaints['product_group'] == prod]
        print('Product group: {p}'.format(p=prod))
        print('Number of documents: {d}'.format(d=len(subset)))
        shuffled_data = shuffle(subset, random_state=RANDOM_SEED)
        document_only = shuffled_data['cleaned_document'].tolist()
        fp = OUTPUT_DIR + prod + '.txt'
        with open(fp, 'w') as f:
            for doc in document_only:
                f.write(doc + "\n")
