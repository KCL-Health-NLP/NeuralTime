
import spacy
from annotated_document import AnnotatedDocument
import pandas as pd
from temporal_extractor import SpacyTemporalExtractor
from corpus import Corpus
import random
from training_spacy import train_model, test_model, trim_entity_spans, merge_intervals
import matplotlib.pyplot as plt
import ast
import numpy as np
from representations import plot_train_test_annotations, plot_annotations, plot_distribution

# ====================================================== Instanciation =================================================

spacy_model = spacy.load("en_core_web_sm")
spacy_extractor = SpacyTemporalExtractor(spacy_model)


# ======================================================= DATA Preparation =============================================

## Annotation format : a dataframe with the following columns : doc, start, end, text, type, value, corpus (?)


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

# to do : create a utility converting cv file to that dictionnary of path


def annotate(annotations, documents):
    ann = []
    for docname in annotations['doc'].unique():

        # extract the annotations
        doc_annotations = annotations[annotations['doc'] == docname][['start', 'end', 'type']].to_numpy()

        ann += [doc_annotations]

    documents['annotations'] = ann
    return documents



def create_data(annotations, file_path_dict):
    docs = []
    for docname in annotations['doc'].unique():

        # extract the annotations
        #doc_annotations = annotations[annotations['doc'] == docname][['start', 'end', 'type']].to_numpy()

        corpus = annotations[annotations['doc'] == docname]['corpus'].unique()

        # extract the text
        f = open(file_path_dict[docname])
        text = f.read()

        # affect to test or train set
        test = False
        if random.random() >= 0.9:
            test = True

        docs += [(docname,  text, corpus, test)]

    return pd.DataFrame(docs, columns = ['docname', 'text', 'corpus', 'test'])


# ============================================= MODEL TRAINING =================================================

# model parameters
spacy_type = False   #choosing which typing to use. True for DATE and TIME types.
other_annotations = False
nb_iter = 55

output_dir = 'C:/Users/LouiseDupuis/Documents/PythonSUTime/models/all_types_model'


# data creation
all_annotations = pd.read_excel('data/all_mtsamples_annotations.xlsx')
file_path_dict = dict([(docname, 'data/corpus/' + docname + '.txt') for docname in all_annotations['doc'].unique()])
documents = pd.read_excel('data/mtsamples_data.xlsx')

# data preprocessing
documents = annotate(all_annotations, documents)
# type conversion
if spacy_type:
    documents['annotations'] = [[(start, end, 'TIME') if label == 'TIME' else (start, end, 'DATE') for (start, end, label) in annotations] for annotations in documents['annotations'].to_numpy()]

# merge intervals : combines overlapping annotations
documents['annotations'] = [merge_intervals(entities) for entities in documents['annotations'].to_numpy()]

train_docs = documents[documents.test == False]
test_docs = documents[documents.test == True]


# Analysis of document distribution

#plot_distribution(documents)
#plot_annotations(all_annotations)
print(plot_train_test_annotations(documents[documents.test == False]))
print(plot_train_test_annotations(documents[documents.test == True]))

# model training
scores = train_model(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type, other_annotations)



# ==========================================  TESTING =======================================================================

model_path = output_dir
# test the saved model
print("Loading from", )
nlp2 = spacy.load(model_path)
test_model(test_docs, nlp2)









'''# trick to get all of the recurrent documents
all_annotations['annotator'] = all_annotations['annotator'].astype(str)
all_annotations['name'] = all_annotations[['annotator', 'corpus', 'doc']].agg('~~'.join, axis=1)    # problem if ~~ appears in a doc name : unlikely, but still
print(all_annotations['name'])
gs_annotations = pd.read_excel('data/mtsamples_annotations.xlsx')

docs = [AnnotatedDocument('data/corpus/' + doc_name.split('~~')[-1] + '.txt') for doc_name in all_annotations['doc'].unique()]

for doc in docs:
    doc.annotate(all_annotations)
'''







# Type based evaluation

# creation of all_annotations

'''
all_annotations = pd.read_excel('data/all_mtsamples_annotations.xlsx')

all_annotations.to_excel('data/all_mtsamples_annotations.xlsx')


all_annotations = pd.DataFrame()
for corpus in corpuses:
    all_annotations = all_annotations.append(annotations_to_extended_standard('data/' + corpus + '_annotations.xlsx', corpus))

all_annotations.to_excel('data/all_mtsamples_annotations.xlsx')

'''


'''for type in types:
    print(type)
    type_annotations = annotations[annotations['type'] == type]
    print(len(type_annotations))
    batch_agreement(annotations['doc'].unique(), type_annotations, spacy_annotations)
    # in doing this, only the false positive are relevant as a lot of annotations would be false negatives for one type'''







'''


    
    
print('DISCHARGE SUMMARIES')
discharge_annotations = annotations_to_gold_standard('data/discharge_annotations.xlsx')
discharge_docs = [AnnotatedDocument('data/discharge-summaries/' + doc_name + '.txt', spacy_model) for doc_name in discharge_annotations['doc'].unique()]
teer_d_expressions = teer_extractor.extract_expressions(discharge_docs)
batch_agreement([doc for doc in discharge_annotations['doc'].unique()],  discharge_annotations, teer_d_expressions)
'''

'''emergency_annotations = annotations_to_gold_standard('data/emergency_annotations.xlsx')
emergency_docs = [AnnotatedDocument('data/emergency/' + doc_name + '.txt', spacy_model) for doc_name in emergency_annotations['doc'].unique()]
spacy_e_expressions = spacy_extractor.extract_expressions(emergency_docs)
print(spacy_e_expressions)
print(emergency_annotations)
batch_agreement([doc for doc in emergency_annotations['doc'].unique()],  emergency_annotations, spacy_e_expressions)
'''

'''

pediatrics_annotations = annotations_to_gold_standard('data/pediatrics_annotations.xlsx')
pediatrics_docs = [AnnotatedDocument('data/pediatrics/' + doc_name + '.txt', spacy_model) for doc_name in pediatrics_annotations['doc'].unique()]
spacy_pe_expressions = spacy_extractor.extract_expressions(pediatrics_docs)
teer_pe_expressions = teer_extractor.extract_expressions(pediatrics_docs)
print(teer_pe_expressions)
print(pediatrics_annotations)
batch_agreement([doc for doc in pediatrics_annotations['doc'].unique()],  pediatrics_annotations, teer_pe_expressions)
'''











