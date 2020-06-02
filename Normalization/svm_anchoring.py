import sklearn
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
from scipy.sparse import hstack
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# need to cross reference between annotations ids and what we know of i2b2 ann

i2b2_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')
all_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')



def normalize_digits(text):

    """
    All number tokens are converted to the token "N"
    :param text: the text to be converted
    :return: the converted text
    """
    all_leter_digits = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                        'twelve']

    new_text = re.sub('\d', 'N', text)

    new_text = ' '.join(['N' if word in all_leter_digits else word for word in new_text.split(' ')])

    return new_text

def initialize_bow(text_documents):
    """
    This function initializes the CountVectorizer object with the vocabulary from the documents
    :param documents: a list of text strings
    :return:
    """

    # normalize digits

    text_documents = [normalize_digits(text) for text in text_documents]

    vectorizer = CountVectorizer()
    vectorizer.fit(text_documents)
    print('Vocabulary size : ' + str(len(vectorizer.get_feature_names())))
    return vectorizer



def get_previous_timexes(id, document_name):

        """
        returns the previous timex (from all the timexes) and previous absolute date or time timex
        :param id: the id of the ri timexe
        :param document_name: the name of the document
        :return: the id of the previous timexe and prvious absolute timexe
        """

        doc_ann = i2b2_timexes[i2b2_timexes.docname == document_name]  # only date and times
        all_doc_ann = all_timexes[all_timexes.docname == document_name]  # all timexes annotations

        ordered_ids = [id for id in doc_ann['id']]
        all_ordered_ids = [id for id in all_doc_ann['id']]

        ordered_ids.sort()
        ordered_ids.sort(key = len)
        index = ordered_ids.index(id)

        all_ordered_ids.sort()
        all_ordered_ids.sort(key=len)
        all_index = all_ordered_ids.index(id)

        # finding the previous timexe id
        if all_index > 0:
                previous_timex_id = all_ordered_ids[all_index -1]
        else:
            previous_timex_id = all_ordered_ids[all_index]

        # finding the previous absolute timexe id
        b = list(zip(doc_ann.id, doc_ann.absolute))
        count = index
        previous_absolute_id = ordered_ids[count]
        while count >0:
            if b[count][1]:
                previous_absolute_id = ordered_ids[count]
                return previous_timex_id, previous_absolute_id
            else:
                count -= 1
        return previous_timex_id, previous_absolute_id



def extract_features(annotations):

    """
    Takes ri timexes annotations in dataframe format and returns the vectors that will be used for the classification

    :param annotations: a dataframe
    :return:
    """
    # initialize tokenizer

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.Defaults.create_tokenizer(nlp)


    # extract the relevant features



    def get_n_token_window(id, document_name, document_text, n = 8):

        ann = all_timexes[(all_timexes.docname == document_name) & (all_timexes.id == id)]
        start = ann['start'].unique()[0]
        end = ann['end'].unique()[0]

        tokenized_text = tokenizer(document_text)
        span = tokenized_text.char_span(start, end)
        token = span.merge()

        window = normalize_digits(tokenized_text[token.i - n: token.i + n].text)
        print(start,end)
        print(token.i)
        print('Token : ' + str(token.text))
        print('Window : ' + str(window))
        print()
        return window





    # get the window of tokens around the expression
    windows = [get_n_token_window(id, docname, text_documents[docname])for id, docname in zip(annotations['TIMEX_id'], annotations['docname'])]
    window_vectors = vectorizer.transform(windows)
    print(window_vectors.shape)

    # extract previous timex and previous absolute timex
    previous_ids = [get_previous_timexes(id, docname) for id, docname in zip(annotations['TIMEX_id'], annotations['docname'])]

    # add previous timex vectors
    print('Previous T')
    previous_timexes = [get_n_token_window(ids[0], docname, text_documents[docname], 1) for ids, docname in zip(previous_ids, annotations.docname)]
    previous_timexes_vectors = vectorizer.transform(previous_timexes)
    print(previous_timexes_vectors.shape)

    # add previous absolute timex vectors
    print('Previous absolute T')
    previous_abs_timexes = [get_n_token_window(ids[1], docname, text_documents[docname], 1) for ids, docname in zip(previous_ids, annotations.docname)]
    previous_abs_timexes_vectors = vectorizer.transform(previous_abs_timexes)
    print(previous_abs_timexes_vectors.shape)

    # concatenate the vectors

    vectors = hstack([window_vectors, previous_timexes_vectors, previous_abs_timexes_vectors])

    # select features

    print('Feature Selection ')
    print(vectors.shape)
    """selector = VarianceThreshold()
    vectors = selector.fit_transform(vectors)"""

    print(vectors.shape)

    return vectors


import random
random.seed(42)

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')
y = ri_original_timexes['Anchor'].to_numpy()

# Train/Test split

train_data, test_data, y_train, y_test = train_test_split(ri_original_timexes, y, test_size=0.15, random_state=0, stratify=y)


print(ri_original_timexes)

# extracting document text

path = '../TimeDatasets/i2b2 Data/Test_data/merged_i2b2/'

text_documents = {}
for filepath in ri_original_timexes['docname'].unique() :
    text_path = filepath + '.txt'
    text = open( path + text_path, 'r').read()
    text_documents[filepath] = text

# initialize the vectorizer
vectorizer = initialize_bow([text for text in text_documents.values()])


# random train/test set

"""train_data = ri_original_timexes[ri_original_timexes.test == False]
test_data = ri_original_timexes[ri_original_timexes.test == True]"""

def prepare_data():
    print()
    print('Data Distribution')

    print(len(ri_original_timexes))

    g = ri_original_timexes.groupby('Anchor').agg(['count'])['docname']
    print(g)

    print( )
    print('Train set : ' + str(len(train_data)) + ' documents')
    print(train_data.groupby('Anchor').agg(['count'])['docname'])
    print()
    print('Test set : ' + str(len(test_data)) + ' documents')
    print(test_data.groupby('Anchor').agg(['count'])['docname'])
    print()



    X_train = extract_features(train_data)
    X_test = extract_features(test_data)


def svm_anchoring():

    anchor_types = ['A', 'D', 'P', 'PA']

    optimized_parameters = dict()
    optimized_parameters['A'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['D'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['P'] = {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['PA'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

    def train_model(anchor_type):
        """
        trains a classification model for one type of anchor
        """

        print()
        print('==================================================================')
        print(anchor_type)
        print()


        y_train_binary = [1 if type == anchor_type else 0 for type in y_train]
        y_test_binary = [1 if type == anchor_type else 0 for type in y_test]

        # Cross Validation

        """# Parameters otpimization
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

        clf = GridSearchCV(
            svm.SVC(), tuned_parameters, scoring='f1')
        clf.fit(X_train, y_train_binary)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()"""

        params = optimized_parameters[anchor_type]
        clf = svm.SVC(C = params['C'], kernel = params['kernel'], gamma= params['gamma'])

        scores = cross_val_score(clf, X_train, y_train_binary, cv=10, scoring = 'f1')
        precisions = cross_val_score(clf, X_train, y_train_binary, cv=10, scoring = 'precision')
        recalls  = cross_val_score(clf, X_train, y_train_binary, cv=10, scoring = 'recall')

        print('Cross validation - f1 scores :')
        print(scores)
        print('Cross validation - Precision :')
        print(precisions)
        print('Cross validation - Recall :')
        print(recalls)

        print('Average Precision : ' + str(sum(precisions) / len(precisions)))
        print('Average Recall : ' + str(sum(recalls) / len(recalls)))
        print('Average F1 : ' + str(sum(scores) / len(scores)))

        print()

        # Train the model on the whole dataset

        clf.fit(X_train, y_train_binary)

        # Test the model

        y_pred = clf.predict(X_test)
        print(y_pred)
        print()

        print(sklearn.metrics.classification_report(y_test_binary, y_pred))

        print('Accuracy :')
        print(clf.score(X_test, y_test_binary))

        fscore = f1_score(y_test_binary, y_pred)
        print('F1 Score ' + str(fscore))

        print()
        print()


    for t in anchor_types:
        train_model(t)



    return None



"""prepare_data()
svm_anchoring()"""