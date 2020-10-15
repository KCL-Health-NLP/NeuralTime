import random
import dataset
import data_analysis
import bert_transformer

import transformers
import pandas as pd
import os

import sys

sys.path.append('/exps/louise/NeuralTime/RI_Annotations')

random.seed(4)
## Model Parameters

do_train = True
max_seq_length = 512
learning_rate = 2e-5
num_train_epochs = 15
gradient_accumulation_steps = 0.9
train_batch_size = 5
fp16 = False

# data parameters
val_prop = 0.2
undersample = 0.5

# load tokenizer
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# load and prepare dataset
inf_train_features = pd.read_excel('DataTables/inference_train_features.xlsx')
#train_features = pd.read_excel('DataTables/train_features.xlsx')

test_features = pd.read_excel('DataTables/test_features.xlsx')
bert_transformer.clinical_bert(inf_train_features, test_features, do_train, learning_rate, num_train_epochs, train_batch_size, val_prop = val_prop, undersample = undersample, multilabel_mode = True, fp16 = False)
