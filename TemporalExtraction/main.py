
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
import random

# ====================================================== Instanciation =================================================

random.seed(42)

# Loading the models
spacy_model = spacy.load("en_core_web_sm")
spacy_extractor = SpacyTemporalExtractor(spacy_model)

mtsamples_model = spacy.load('models/all_types_model/all_types_model_on_all_data/')
original_mtsamples_model = spacy.load('models/all_types_model/all_types_model_on_all_data/') # this model won't be retrained

# current datetime to name the exported files
currentDT = str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace(' ', '_')
print(currentDT)


# ======================================================= DATA Preparation =============================================

# We need the data as : an annotation dataframe, a documents dataframe, and,
# if training, to divide the documents between train_docs and test_docs
# the load_data function can output these if given an annotations dataframe and the paht to the files (see required format in data_preparation)



all_annotations, documents, train_docs, test_docs = load_mt_samples()  # MT samples data
#cris_annotations, cris_documents, cris_train_docs, cris_test_docs = load_data(...) # CRIS data

i2b2_annotations, i2b2_documents = load_data('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')
test_i2b2_docs = i2b2_documents[i2b2_documents.test == True]
train_i2b2_docs = i2b2_documents[i2b2_documents.test == False]
print(i2b2_documents)
# ============================================= MODEL TRAINING =================================================

# model parameters
spacy_type = False   #choosing which typing to use. True for DATE and TIME types.
other_annotations = False
nb_iter = 2


if not spacy_type:
    output_dir = 'models/all_types_model_' + currentDT
else:
    output_dir = 'models/spacy_types_model_' + currentDT

i2b2_output_dir = 'models/i2b2_' + currentDT

# Analysis of document distribution
# TO DO : adapt the representation functions to all documents

plot_annotations(i2b2_annotations, 'representation_and_evaluation/i2b2_annotation_distribution.xlsx')
#plot_distribution(i2b2_documents, 'representation_and_evaluation/i2b2_document_distribution.xlsx')
print('TEST')
print(test_i2b2_docs)
print(plot_train_test_annotations(test_i2b2_docs))
print()
print('TRAIN')
print(train_i2b2_docs)
print(plot_train_test_annotations(train_i2b2_docs))

# Model Training

scores = train_model(mtsamples_model, train_i2b2_docs, test_i2b2_docs, nb_iter, i2b2_output_dir, spacy_type)


# ==========================================  TESTING =======================================================================

#model_path = output_dir + '/all_types_model_final/'
# test the saved model
#print("Loading from", )
#nlp2 = spacy.load(model_path)
#test_model(test_docs, nlp2)

test_model(test_i2b2_docs, original_mtsamples_model)
#plot_training(pd.read_excel('all_types_model_2fold.xlsx'))
#plot_training(pd.read_excel('all_types_model_on_all_data.xlsx'))

















