
import spacy
import pandas as pd
import numpy as np
from temporal_extractor import SpacyTemporalExtractor
from utilities import merge_intervals
from training_spacy import train_model, test_model, trim_entity_spans, train_model_cross_validation
from representations import plot_train_test_annotations, plot_annotations, plot_distribution, plot_training
from data_preparation import load_mt_samples, load_data
import datetime
import os

# ====================================================== Instanciation =================================================

spacy_model = spacy.load("en_core_web_sm")
spacy_extractor = SpacyTemporalExtractor(spacy_model)

currentDT = str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace(' ', '_')
print(currentDT)

local_path = 'C:/Users/LouiseDupuis/Documents/NeuralTime'   # to change to your path

# ======================================================= DATA Preparation =============================================

all_annotations, documents, train_docs, test_docs = load_mt_samples()  # MT samples data

#cris_annotations, cris_documents, cris_train_docs, cris_test_docs = load_data(...) # CRIS data

# ============================================= MODEL TRAINING =================================================

# model parameters
spacy_type = False   #choosing which typing to use. True for DATE and TIME types.
other_annotations = False
nb_iter = 1


if not spacy_type:
    output_dir = 'models/all_types_model_' + currentDT
else:
    output_dir = 'models/spacy_types_model_' + currentDT

os.mkdir(output_dir)

# Analysis of document distribution

plot_distribution(documents)

#plot_annotations(all_annotations)
#print(plot_train_test_annotations(documents[documents.test == False]))
#print(plot_train_test_annotations(documents[documents.test == True]))

# model training
#scores = train_model_cross_validation(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type, nb_folds=5)
scores = train_model(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type)

# ==========================================  TESTING =======================================================================

model_path = output_dir + '/all_types_model_final/'
# test the saved model
print("Loading from", )
nlp2 = spacy.load(model_path)
test_model(test_docs, nlp2)

#plot_training(pd.read_excel('all_types_model_2fold.xlsx'))
#plot_training(pd.read_excel('all_types_model_on_all_data.xlsx'))






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


















