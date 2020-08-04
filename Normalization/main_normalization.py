
import random
import dataset
import map_custom_annotations
import data_analysis
import bert_transformer
import svm_anchoring
import embeddings
random.seed(42)
import transformers
import pandas as pd
import os

import sys

sys.path.append('/exps/louise/NeuralTime/RI_Annotations')


# ==================== Training Anchor Classification models with the original data
"""
ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')

date_and_time = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')  # for now, original filtering
all_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')
# we need to compute all anchor dates for each case (for example when previous absolute timex = previous timex)
anchors = ['Admission_date', 'Discharge_date', 'Previous_TIMEX', 'Previous_absolute_Timex']
anchors_dict = dict(zip(anchors, ([],[],[],[])))

for row in ri_original_timexes.to_dict('records'):
    anchor = row['Anchor']
    if anchor == 'A':
        anchor_value = row['Admission_date']
    if anchor == 'D':
        anchor_value = row['Discharge_date']
    if anchor == 'P':
        anchor_value = row['Previous_TIMEX']
    if anchor == 'PA':
        anchor_value = row['Previous_absolute_Timex']
    if anchor == 'N':
        anchor_value = ''
    for anchor in anchors:
        if anchor_value == row[anchor]:
            anchors_dict[anchor] += [True]
        else :
            anchors_dict[anchor] += [False]

ri_original_timexes['A'] = anchors_dict['Admission_date']
ri_original_timexes['D'] = anchors_dict['Discharge_date']
ri_original_timexes['P'] = anchors_dict['Previous_TIMEX']
ri_original_timexes['PA'] = anchors_dict['Previous_absolute_Timex']

ri_original_timexes['After'] = [anchor_rel == 'A' for anchor_rel in ri_original_timexes['Relation_to_anchor']]
ri_original_timexes['E'] = [anchor_rel == 'E' for anchor_rel in ri_original_timexes['Relation_to_anchor']]
ri_original_timexes['B'] = [anchor_rel == 'B' for anchor_rel in ri_original_timexes['Relation_to_anchor']]





#mimicII_vectorizer = embeddings.MimicIIEmbeddingVectorizer()
models = svm_anchoring.svm_anchoring(ri_original_timexes, date_and_time, all_timexes, vectorizer = 'default', normalize_numbers = False)
"""

# ===============================  mapping our custom data to the original format + training
"""


# test, only on two files

filepaths = [ '../RI_Annotations/AnnotatedData/' + docname  for docname in os.listdir('../RI_Annotations/AnnotatedData') ]

print('Number of documents :' + str(len(filepaths)))
anchorlinks, timexes = map_custom_annotations.annotated_files_to_dataframe(filepaths)

timexes.to_excel('annotated_timexes.xlsx')

mapped_data = map_custom_annotations.custom_to_standard(anchorlinks, timexes, all_timexes)
print(mapped_data)
data_analysis.analysis_mapped_custom_data(mapped_data)
models = svm_anchoring.svm_anchoring(mapped_data, date_and_time, all_timexes, vectorizer = 'default', normalize_numbers = True)


"""
# ======================================= Bert model training

#bert_data.generate_input(out_file='train_inputs_2.xlsx')

#examples = bert_data.generate_input(out_file= 'train_inputs.xlsx')
#print(examples)

## Model Parameters

do_train = True
max_seq_length = 512
learning_rate = 2e-5
num_train_epochs = 10
gradient_accumulation_steps = 0.9
train_batch_size = 5
warmup_proportion = 0.1

# load tokenizer
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

#bert_model = transformers.BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# load and prepare dataset
data = dataset.AnnotatedDataset(inference = True)

#train_features = bert_transformer.convert_examples_to_features( train_examples, label_list, max_seq_length, bert_tokenizer)
#data.generate_inputs(type = 'test', out_path= 'test_inputs.xlsx')

#data.convert_to_tuples(out_path = 'tuple_df.xlsx', inference=False)
#train_examples = data.generate_input(type = 'train', out_file = 'train_inputs.xlsx')
#train_features = bert_transformer.convert_examples_to_features(data.get_examples(), max_seq_length, bert_tokenizer, out_path = 'train_features.xlsx' )

#test_features = bert_transformer.convert_examples_to_features(data.get_examples(type = 'test'), max_seq_length, bert_tokenizer, out_path = 'test_features.xlsx')
inf_train_features = pd.read_excel('DataTables/inference_train_features.xlsx')
#train_features = pd.read_excel('DataTables/train_features.xlsx')

test_features = pd.read_excel('DataTables/test_features.xlsx')
bert_transformer.clinical_bert(inf_train_features, test_features, do_train, learning_rate, num_train_epochs, train_batch_size, undersample = True, multilabel_mode = True, fp16 = False)

