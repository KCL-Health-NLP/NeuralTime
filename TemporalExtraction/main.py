
import spacy
import pandas as pd
import numpy as np
from temporal_extractor import SpacyTemporalExtractor
from utilities import merge_intervals
from training_spacy import train_model, trim_entity_spans, train_model_cross_validation
from representations import plot_train_test_annotations, plot_annotations, plot_distribution, plot_training
from data_preparation import load_mt_samples, load_data
from apply_model import test_model
import datetime
import os

# ====================================================== Instanciation =================================================

# Loading the models
spacy_model = spacy.load("en_core_web_sm")
spacy_extractor = SpacyTemporalExtractor(spacy_model)

mtsamples_model = spacy.load('models/all_types_model/all_types_model_on_all_data/')

# current datetime to name the exported files
currentDT = str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace(' ', '_')
print(currentDT)


# ======================================================= DATA Preparation =============================================

# We need the data as : an annotation dataframe, a documents dataframe, and,
# if training, to divide the documents between train_docs and test_docs
# the load_data function can output these if given an annotations dataframe and the paht to the files (see required format in data_preparation)



all_annotations, documents, train_docs, test_docs = load_mt_samples()  # MT samples data
#cris_annotations, cris_documents, cris_train_docs, cris_test_docs = load_data(...) # CRIS data

i2b2_annotations, i2b2_documents = load_data('../TimeDatasets/i2b2 Data/i2b2_training_annotations.xlsx')

print(i2b2_documents)
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

# TO DO : adapt the representation functions to all documents

#plot_annotations(all_annotations)
#print(plot_train_test_annotations(documents[documents.test == False]))
#print(plot_train_test_annotations(documents[documents.test == True]))

# Model Training
#scores = train_model_cross_validation(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type, nb_folds=5)
#scores = train_model(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type)

# ==========================================  TESTING =======================================================================

#model_path = output_dir + '/all_types_model_final/'
# test the saved model
#print("Loading from", )
#nlp2 = spacy.load(model_path)
#test_model(test_docs, nlp2)

test_model(test_docs, mtsamples_model)
test_model(i2b2_documents, mtsamples_model)
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


















