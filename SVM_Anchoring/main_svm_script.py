import random
import dataset
import map_custom_annotations
import data_analysis
import svm_anchoring
import embeddings
random.seed(42)
import pandas as pd
import os

import sys

date_and_time = pd.read_excel('DataTables/date_and_time.xlsx')  # all date and time i2b2 timexes
all_timexes = pd.read_excel('DataTables/all_timexes_sectimes.xlsx') # all i2b2 timexes, including section times


# ==================== Training Anchor Classification models with the original data


ri_2015_timexes = pd.read_excel('ri_2015_modified.xlsx')

#mimicII_vectorizer = embeddings.MimicIIEmbeddingVectorizer() # tried to use embeddings # bad results when using embeddings
models = svm_anchoring.svm_anchoring(ri_2015_timexes, date_and_time, all_timexes, vectorizer = 'default', normalize_numbers = False)

# ===============================  Extracting our annotations ================== =======================================


filepaths = [ '../RI_Annotations/AnnotatedData/' + docname  for docname in os.listdir('../RI_Annotations/AnnotatedData') ]

print('Number of documents :' + str(len(filepaths)))
anchorlinks, annotated_timexes = map_custom_annotations.annotated_files_to_dataframe(filepaths)  # extracting annotated timexes and anchorlinks

anchorlinks.to_excel('DataTables/anchorlinks.xlsx')
annotated_timexes.to_excel('DataTables/annotated_timexes.xlsx')

# ===============================Mapping our custom annotations to Sun et al's format ==================================

anchorlinks = pd.read_excel('DataTables/anchorlinks.xlsx')
annotated_timexes = pd.read_excel('DataTables/annotated_timexes.xlsx')

mapped_data = map_custom_annotations.custom_to_standard(anchorlinks, annotated_timexes, all_timexes)
print(mapped_data)
mapped_data.to_excel('mapped_data.xlsx')

# ========================================Training on mapped data ======================================================

mapped_data = pd.read_excel('mapped_data.xlsx')
#data_analysis.analysis_mapped_custom_data(mapped_data)

models = svm_anchoring.svm_anchoring(mapped_data, annotated_timexes, all_timexes, vectorizer = 'default', normalize_numbers = True)

