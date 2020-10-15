
import random
import dataset
import bert_transformer
random.seed(42)

import sys


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

