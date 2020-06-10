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



def get_previous_timexes(id, document_name, date_and_time, all_timexes):

        """
        returns the previous timex (from all the timexes) and previous absolute date or time timex
        :param id: the id of the ri timexe
        :param document_name: the name of the document
        :return: the id of the previous timexe and prvious absolute timexe
        """

        doc_ann = date_and_time[date_and_time.docname == document_name]  # only date and times
        all_doc_ann = all_timexes[all_timexes.docname == document_name]  # all timexes annotations


        # ordering ids as the timexes appear in the document
        start_ids = [(start, id, abs) for start, id, abs in zip(doc_ann['start'], doc_ann['id'], doc_ann['absolute'])]
        all_ordered_ids = [(start,id) for start,id in zip(all_doc_ann['start'], all_doc_ann['id'])]

        start_ids.sort()
        all_ordered_ids.sort()


        ordered_ids = [s[1] for s in start_ids]
        all_ordered_ids = [s[1] for s in all_ordered_ids]


        index = ordered_ids.index(id)
        all_index = all_ordered_ids.index(id)

        # finding the previous timexe id
        if all_index > 0:
                previous_timex_id = all_ordered_ids[all_index -1]
        else:
            previous_timex_id = all_ordered_ids[all_index]

        # finding the previous absolute timexe id

        if index > 0:
            count = index - 1
            previous_absolute_id = ordered_ids[count]
            while count > 0:
                if start_ids[count][2]:
                    previous_absolute_id = ordered_ids[count]
                    return previous_timex_id, previous_absolute_id
                else:
                    count -= 1
        else :
            previous_absolute_id = ordered_ids[index]

        return previous_timex_id, previous_absolute_id



def extract_features(annotations, text_documents, vectorizer, date_and_time, all_timexes):

    """
    Takes ri timexes annotations in dataframe format and returns the vectors that will be used for the classification

    :param annotations: a dataframe with the following format :
    :param text_documents : a dictionnary with docnames as keys and document text as value
    :param vectorizer : tht vectorizer (already fitted)
    :return: vectors : a sparse matrix of feature vectors
            previous_ids : the ids for previous timexes and previous absolute timexes
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
    previous_ids = [get_previous_timexes(id, docname, date_and_time, all_timexes) for id, docname in zip(annotations['TIMEX_id'], annotations['docname'])]

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

    return vectors, previous_ids







def svm_anchoring(data, date_and_time, all_timexes, path= '../TimeDatasets/i2b2 Data/all_data/'):


    """
     This function takes a dataframe with the anotations and their anchor and anchor relation, and performs training and
     testing of the models, on anchor date and anchor relation

     :param data:
     :param date_and_time : the dataframe containing all date and time timexes (careful of the filtering)
     :param all_timexes : all the original i2b2 timexes annotation (pre ri annotations)
     :param path : path to the xml files for the documents contained in data.docnames
     :return:
     """

    y_anchor = data['Anchor'].to_numpy()
    y_relation = data['Relation_to_anchor'].to_numpy()
    y_relation = [ 'After' if r == 'A' else r for r in y_relation]


    # divide into train/test sets

    try :
        train_data = data[data.test == False]
        test_data = data[data.test == True]  # boolean or strings ?
        y_anchor_train = train_data['Anchor'].to_numpy()
        y_anchor_test = test_data['Anchor'].to_numpy
        y_relation_train = train_data['Relation_to_anchor'].to_numpy()
        y_relation_test = test_data['Relation_to_anchor'].to_numpy
    except Exception as e:
        print(e)
        train_data, test_data, y_anchor_train, y_anchor_test, y_relation_train, y_relation_test = train_test_split(data, y_anchor, y_relation, test_size=0.15, random_state=0, stratify=y_anchor)


    # create text_document
    text_documents = {}
    for filepath in data['docname'].unique():
        text_path = filepath + '.txt'
        text = open(path + text_path, 'r').read()
        text_documents[filepath] = text
    print()
    print('Data Distribution')

    # describe data distribution

    print(str(len(data)) + ' documents')
    print()

    g = data.groupby('Anchor').agg(['count'])['docname']
    print(g)

    print()
    print('Train set : ' + str(len(train_data)) + ' documents')
    print(train_data.groupby('Anchor').agg(['count'])['docname'])
    print()
    print('Test set : ' + str(len(test_data)) + ' documents')
    print(test_data.groupby('Anchor').agg(['count'])['docname'])
    print()

    # initialize the vectorizer
    vectorizer = initialize_bow([text for text in text_documents.values()])

    # extract feature vectors

    X_train, previous_ids_train = extract_features(train_data, text_documents, vectorizer, date_and_time, all_timexes)
    X_test, previous_ids_test = extract_features(test_data, text_documents, vectorizer, date_and_time, all_timexes)



    # training models

    anchor_types = ['A', 'D', 'P', 'PA']

    optimized_parameters = dict()
    optimized_parameters['A'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['D'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['P'] = {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['PA'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    optimized_parameters['B'] = {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
    optimized_parameters['E'] = {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'}
    optimized_parameters['After'] = {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}


    def train_model(anchor_type, y_train, y_test, optimize_params = False):

        """
        trains a classification model for one type of anchor
        """

        print()
        print('==================================================================')
        print(anchor_type)
        print()

        y_train_binary = [1 if type == anchor_type else 0 for type in y_train]
        y_test_binary = [1 if type == anchor_type else 0 for type in y_test]

        print('y train (binary) :')
        print(y_train_binary)
        print(sum(y_train_binary))
        print('y test (binary) :')
        print(y_train_binary)
        print(sum(y_test_binary))

        # for cases where the previous timex and the previous absolute timex are the same :
        if anchor_type == 'P' or anchor_type == 'PA':
            p_and_pa_train = [(p == pa) for p,pa in previous_ids_train]
            p_and_pa_test = [(p == pa) for p, pa in previous_ids_test]
            for i in range(len(p_and_pa_train)):
                if p_and_pa_train[i]:
                    y_train_binary[i] = 1
            for i in range(len(p_and_pa_test)):
                if p_and_pa_test[i]:
                    y_test_binary[i] = 1
            print('Modified if P = PA' )
            print(y_train_binary)
            print(y_test_binary)
        print()

        # Cross Validation

        # Parameters otpimization

        if optimize_params :
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
            print()

        try:
            params = optimized_parameters[anchor_type]
            clf = svm.SVC(C = params['C'], kernel = params['kernel'], gamma= params['gamma'])
        except KeyError :
            print('No optimized parameters')
            clf = svm.SVC()

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

        print('y pred :')
        print(y_pred)
        print()

        print(sklearn.metrics.classification_report(y_test_binary, y_pred))

        print('Accuracy :')
        print(clf.score(X_test, y_test_binary))

        fscore = f1_score(y_test_binary, y_pred)
        print('F1 Score ' + str(fscore))

        print()
        print()

        return clf


    models = []
    for t in anchor_types:
        models += [train_model(t, y_anchor_train, y_anchor_test)]

    relations = ['B', 'E', 'After']
    for r in relations:
        models += [train_model(r, y_relation_train, y_relation_test, optimize_params= False)]


    return models

