
import random
import pandas as pd
import svm_anchoring
import map_custom_annotations
import data_analysis
import os
import embeddings

import RI_Annotations.dataset as dataset

import bert_transformer
random.seed(42)


"""
# training anchor classififation models with the original data

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')

date_and_time = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')  # for now, original filtering
all_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')


mimicII_vectorizer = embeddings.MimicIIEmbeddingVectorizer()
#models = svm_anchoring.svm_anchoring(ri_original_timexes, date_and_time, all_timexes, vectorizer = mimicII_vectorizer, normalize_numbers = False)


# mapping our custom data to the original format

# test, only on two files

filepaths = [ '../RI_Annotations/AnnotatedData/' + docname  for docname in os.listdir('../RI_Annotations/AnnotatedData') ]

print('Number of documents :' + str(len(filepaths)))
anchorlinks, timexes = map_custom_annotations.annotated_files_to_dataframe(filepaths)

timexes.to_excel('annotated_timexes.xlsx')

mapped_data = map_custom_annotations.custom_to_standard(anchorlinks, timexes, all_timexes)

data_analysis.analysis_mapped_custom_data(mapped_data)
models = svm_anchoring.svm_anchoring(mapped_data, date_and_time, all_timexes, vectorizer = 'default', normalize_numbers = True)
"""

data = dataset.AnnotatedDataset()
bert_train_data = bert_transformer.BertDataset(data, 'DataTables/train_inputs.xlsx')


#bert_data.generate_input(out_file='train_inputs_2.xlsx')

#examples = bert_data.generate_input(out_file= 'train_inputs.xlsx')
#print(examples)

## Model Parameters

do_train = True
max_seq_length = 512
learning_rate = 0.00005
num_train_epochs = 3
gradient_accumulation_steps = 1.1
train_batch_size = 32
warmup_proportion = 0.1

bert_transformer.clinical_bert(do_train, max_seq_length, learning_rate, num_train_epochs, gradient_accumulation_steps, train_batch_size, warmup_proportion)

