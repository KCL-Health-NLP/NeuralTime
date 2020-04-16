import spacy
import pandas as pd
import numpy as np
import random
from utilities import merge_intervals
import pickle
import ntpath


## Annotation format : a dataframe with the following columns : doc, start, end, text, type, value, corpus


def mt_samples_to_standard_data():
    corpus_names = ['discharge-summaries', 'psychiatry-psychology', 'emergency', 'pediatrics']

    all_annotations = pd.DataFrame()
    for corpus  in corpus_names:
        raw_annotations_path = 'data/' + corpus + '_annotations.xlsx'
        raw_annotations = pd.read_excel(raw_annotations_path,
                                        names=['doc', 'status', 'text1', 'text2', 'type1', 'type2', 'value1', 'value2',
                                               'start1', 'end1', 'start2', 'end2', 'context', 'corpus'])

        # first annotator & agreement
        annotations1 = raw_annotations[((raw_annotations.status == 'TP') | (raw_annotations.status == 'FN')) ][['doc', 'start1', 'end1', 'type1', 'text1', 'value1', 'corpus']].rename(columns = {'start1' : 'start', 'end1' : 'end', 'text1' : 'text', 'type1' : 'type', 'value1' : 'value'})
        # second annotator
        annotations2 = raw_annotations[raw_annotations.status == 'FP'][['doc', 'start2', 'end2', 'type2', 'text2', 'value2', 'corpus']].rename(columns = {'start2' : 'start', 'end2' : 'end', 'text2' : 'text', 'type2' : 'type','value2' : 'value'})

        annotations = pd.concat([annotations1, annotations2], axis=0)

        annotations['doc'] = [doc.strip() for doc in annotations['doc']] #remove trailing whitespaces

        # selecting entities types + setting floats to int
        annotations = annotations[annotations.type != 'other'].astype({'start' : int, 'end' : int})

        # converting types to uppercase to match spacy's types
        annotations['type'] = annotations['type'].apply(lambda x : x.upper())

        annotations['corpus'] = [corpus for i in range(len(annotations))]

        all_annotations = all_annotations.append(annotations)

    return all_annotations


# to initialise the data,you must give a list of standard annotations (a different dataframe for each annotator) must be provided
# as well as a dictionnary mapping the path of the doc to the info of the standard_annotations

# to do : create a utility converting csv file to a dictionnary of path


def annotate(annotations, documents):
    ann = []
    for docname in annotations['docname'].unique():

        # extract the annotations
        doc_annotations = annotations[annotations['docname'] == docname][['start', 'end', 'type']].to_numpy()

        ann += [doc_annotations]

    documents['annotations'] = ann
    return documents



def create_data(annotations, file_path_dict):
    docs = []
    for docname in annotations['docname'].unique():

        # extract the annotations
        #doc_annotations = annotations[annotations['doc'] == docname][['start', 'end', 'type']].to_numpy()

        try:
            corpus = annotations[annotations['docname'] == docname]['corpus'].unique()
        except Exception as e:
            corpus = ['unknown' for i in range(len(annotations))]



        # extract the text
        f = open(file_path_dict[docname])
        text = f.read()

        # affect to test or train set
        test = False
        if random.random() >= 0.9:
            test = True

        docs += [(docname,  text, corpus, test)]

    return pd.DataFrame(docs, columns = ['docname', 'text', 'corpus', 'test'])



def load_mt_samples():

    # data creation
    all_annotations = pd.read_excel('../TimeDatasets/MTSamples/all_mtsamples_annotations.xlsx')
    file_path_dict = dict([(docname, '../TimeDatasets/MTSamples/corpus/' + docname + '.txt') for docname in all_annotations['docname'].unique()])
    documents = pd.read_excel('../TimeDatasets/MTSamples/mtsamples_data.xlsx')

    # data preprocessing
    documents = annotate(all_annotations, documents)
    # type conversion
    spacy_type = False
    if spacy_type:
        documents['annotations'] = [[(start, end, 'TIME') if label == 'TIME' else (start, end, 'DATE') for (start, end, label) in annotations] for annotations in documents['annotations'].to_numpy()]

    # merge intervals : combines overlapping annotations
    documents['annotations'] = [merge_intervals(entities) for entities in documents['annotations'].to_numpy()]

    train_docs = documents[documents.test == False]
    test_docs = documents[documents.test == True]

    return all_annotations, documents, train_docs, test_docs




def load_data(annotations_path, file_path_dict_path = None):

    # data creation
    try:
        all_annotations = pd.read_excel(annotations_path)
    except Exception as e:
        print(e)
        all_annotations = pd.read_csv(annotations_path)

    if file_path_dict_path:
        f = open(file_path_dict_path, 'rb')
        file_path_dict = pickle.load(f)
    else:
        # if annotations has a "text_path" column
        try  :

            file_path_dict = dict(set(list(zip(all_annotations.docname, all_annotations.text_path))))
            print(file_path_dict)
        except Exception as e:
            print('Error : ')
            print(e)
            if file_path_dict_path is None:
                print('You need to provide the file path for every document, either in the annotations (columns "text_path") or as a dictionnary')
                return None

    documents = create_data(annotations=all_annotations, file_path_dict=file_path_dict)

    # data preprocessing
    documents = annotate(all_annotations, documents)
    # type conversion
    spacy_type = False
    if spacy_type:
        documents['annotations'] = [
            [(start, end, 'TIME') if label == 'TIME' else (start, end, 'DATE') for (start, end, label) in annotations]
            for annotations in documents['annotations'].to_numpy()]

    # merge intervals : combines overlapping annotations
    documents['annotations'] = [merge_intervals(entities) for entities in documents['annotations'].to_numpy()]

    return all_annotations, documents



def load_raw_data(path_list):
    text_list = [open(path, 'r').read() for path in path_list]
    data = pd.DataFrame
    data['doc'] = [ ntpath.basename(path) for path in path_list]
    data['text'] = text_list
    return data
