
import spacy
import pandas as pd
import numpy as np
from temporal_extractor import SpacyTemporalExtractor
from training_spacy import train_model, test_model, trim_entity_spans, merge_intervals
from representations import plot_train_test_annotations, plot_annotations, plot_distribution
from data_preparation import load_mt_samples, load_data
# ====================================================== Instanciation =================================================

spacy_model = spacy.load("en_core_web_sm")
spacy_extractor = SpacyTemporalExtractor(spacy_model)


# ======================================================= DATA Preparation =============================================

all_annotations, documents, train_docs, test_docs = load_mt_samples()

# ============================================= MODEL TRAINING =================================================

# model parameters
spacy_type = False   #choosing which typing to use. True for DATE and TIME types.
other_annotations = False
nb_iter = 55

if not spacy_type:
    output_dir = 'C:/Users/LouiseDupuis/Documents/PythonSUTime/models/all_types_model'
else:
    output_dir = 'C:/Users/LouiseDupuis/Documents/PythonSUTime/models/spacy_types_model'


# Analysis of document distribution

#plot_distribution(documents)
#plot_annotations(all_annotations)
#print(plot_train_test_annotations(documents[documents.test == False]))
#print(plot_train_test_annotations(documents[documents.test == True]))

# model training
#scores = train_model(spacy_model, train_docs, test_docs, nb_iter, output_dir, spacy_type, other_annotations)



# ==========================================  TESTING =======================================================================

model_path = 'models/all_types_model/all_types_model_on_all_data'
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











