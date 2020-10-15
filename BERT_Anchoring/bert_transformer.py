import os

import sklearn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
import pandas as pd
import transformers
import numpy as np
from dataset import AnnotatedDataset
from tqdm import trange, tqdm
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
import ast
import random
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix
from bert_for_multilabel_classification import BertForMultiLabelSequenceClassification
import scipy
import dataset


from datetime import datetime



"""
Utility functions to train the BERT model 
"""


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def map_label(label_string):
    """
    Utility function to map the labels from input dataframe into a format suited to Bert classifier
    :return:
    """
    # cast to list
    label = ast.literal_eval(label_string)

    # convert to a multiclassification output
    label_list = [int(not bool(label[0])), label[1], label[2], label[3]]

    return label_list



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, max_seq_length, tokenizer, out_path):
    """
    Converts data with the text input to features that can then be passed to the model (tokens are identified by their id in the
    tokenizer's vocabulary)

    :param examples: list of examples, of class InputExample (see dataset.py)
    :param max_seq_length:
    :param tokenizer:
    :param out_path:
    :return:
    """

    #label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    features_list = []
    max_len = 0
    for  idx, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tuple_id = example.tuple_id

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if seq_len > max_len:
            max_len = seq_len
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        print(example.label)
        print(type(example.label))
        label_ = map_label(example.label)
        label_id = label_.index(1)
        print(label_id)
        if idx < 3:
            print("*** Example ***")
            print("guid: %s" % (example.tuple_id))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))


        features_list.append([tuple_id, input_ids, input_mask, segment_ids, label_id, example.label])

    print('Max Sequence Length: %d' % max_len)


    features_df = pd.DataFrame(features_list, columns=['tuple_id', 'input_id', 'input_mask', 'segment_id', 'label_id', 'label'])
    print('FEATURES ')
    print(features_df)


    if out_path is not None:
        features_df.to_excel(out_path)
    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def clinical_bert(train_features, test_set_features, do_train, learning_rate, num_train_epochs, train_batch_size, val_prop = 0.1, undersample = 0.5, multilabel_mode = False, fp16 = True):

    """

    :param train_features: the generated features for training the model
    :param test_set_features: the features for the test set
    :param do_train: if the model will be trained or only evaluated on the test set
    :param learning_rate: the model's learning rate
    :param num_train_epochs: the number of epoch to train the model for
    :param train_batch_size: the batch size for training
    :param val_prop: the proportion to keep for validation set, between 0 and 1
    :param undersample:the proportion to decrease the number of training examples from the dominant class (the "not an anchor" class)
    :param multilabel_mode: wether to use a 4 case classification or a multilabel classification
    :param fp16: wether or not to use half precision for the calculations
    :return:
    """

    # initializing the models
    if not multilabel_mode:
        bert_model = transformers.BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model.classifier = torch.nn.Linear(768, 4)
    else:
        bert_model = BertForMultiLabelSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        bert_model.classifier = torch.nn.Linear(768, 4)

    # change the output layer

    print(bert_model.classifier)
    bert_model.num_labels = 4

    if fp16:
        bert_model.half()

    # preparing train and test data

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    bert_model.to(device)

    # Prepare optimizer


    optimizer = AdamW(bert_model.parameters(),
                      lr= learning_rate,
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )



    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if do_train:

        random.seed(42)

        # divide between train and validation set

        # undersample the training set - keeping only part of the most dominant class
        if undersample:
            undersample_mask = [(random.random() > 1 - undersample) if ast.literal_eval(label) == [0,0,0,0] else True for label in
                                train_features['label']]
            train_features = train_features[undersample_mask]


        if val_prop > 0:
            val_mask = [(random.random() >= 1 - val_prop) for i in range(len(train_features))]
            train_set_features = train_features[[not f for f in val_mask]]
            val_set_features = train_features[val_mask]
        else:
            train_set_features = train_features





        # =================== Data Analysis
        # print a representation of the train, validation and test set before training


        def data_analysis(feature_set):

            total_nb = len(feature_set)
            nb_anchor_dates = sum([ast.literal_eval(label)[0] for label in feature_set['label']])
            nb_before = sum([ast.literal_eval(label)[1] for label in feature_set['label']])
            nb_equal = sum([ast.literal_eval(label)[2] for label in feature_set['label']])
            nb_after = sum([ast.literal_eval(label)[3] for label in feature_set['label']])
            print(str(total_nb) + ' total examples ')
            print(' Number of examples that are anchor dates : ' + str(nb_anchor_dates))
            print('Before ', nb_before)
            print('Equal ', nb_equal)
            print('After ', nb_after)
            print()

        print('TRAIN SET :')
        data_analysis(train_set_features)
        print('VALIDATION SET :')
        data_analysis(val_set_features)
        print('TEST SET :')
        data_analysis(test_set_features)






        print("***** Running training *****")
        print("  Num examples = %d", len(train_set_features))
        print("  Batch size = %d", train_batch_size)

        all_input_ids = torch.tensor([ast.literal_eval(f) for f in train_set_features['input_id'].values], dtype=torch.long)
        all_input_mask = torch.tensor([ast.literal_eval(f) for f in train_set_features['input_mask'].values], dtype=torch.long)
        all_segment_ids = torch.tensor([ast.literal_eval(f) for f in train_set_features['segment_id'].values], dtype=torch.long)
        if not multilabel_mode:
            all_label_ids = torch.tensor([f for f in train_set_features['label_id'].values], dtype=torch.long)
        else:
            all_label_ids = torch.tensor([ast.literal_eval(f) for f in train_set_features['label'].values], dtype=torch.long)


        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= train_batch_size)

        if val_prop > 0:

            val_input_ids = torch.tensor([ast.literal_eval(f) for f in val_set_features['input_id'].values],
                                         dtype=torch.long)
            val_input_mask = torch.tensor([ast.literal_eval(f) for f in val_set_features['input_mask'].values],
                                          dtype=torch.long)
            val_segment_ids = torch.tensor([ast.literal_eval(f) for f in val_set_features['segment_id'].values],
                                           dtype=torch.long)

            if not multilabel_mode:
                val_label_ids = torch.tensor([f for f in val_set_features['label_id'].values], dtype=torch.long)
            else:
                val_label_ids = torch.tensor([ast.literal_eval(f) for f in val_set_features['label'].values], dtype=torch.long)

            val_data = TensorDataset(val_input_ids, val_input_mask, val_segment_ids, val_label_ids)
            val_sampler = RandomSampler(val_data)
            validation_dataloader = DataLoader(val_data, sampler=None, batch_size=train_batch_size)

            val_metrics = []  # keeping track of the metrics for each epoch

        for _ in trange(int(num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # ========================================
            #               Training
            # ========================================

            bert_model.train()  # put the model in train mode


            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):


                # clear gradients ??
                #bert_model.zero_grad()

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                try:
                    label_ids = label_ids.view((5, 4))
                except Exception as e:
                    print(e)
                    print(label_ids)
                    label_ids.view((-1,4))



                loss, logits = bert_model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids, labels = label_ids)

                #print(loss)

                tr_loss += loss.item()

                # Actual Training
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.



            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            def run_eval(dataloader, set_features):
                print("")
                print("Running Validation...")

                #t0 = time.time()

                # Put the model in evaluation mode--the dropout layers behave differently
                # during evaluation.
                bert_model.eval()

                # Tracking variables
                total_eval_metrics = []
                total_eval_loss = 0
                nb_eval_steps = 0

                val_labels = []
                val_pred = []
                val_pred_int = []




                # Evaluate data for one epoch
                for batch in dataloader:
                    # Unpack this training batch from our dataloader.
                    #
                    # As we unpack the batch, we'll also copy each tensor to the GPU using
                    # the `to` method.
                    #
                    # `batch` contains three pytorch tensors:
                    #   [0]: input ids
                    #   [1]: attention masks
                    #   [2]: labels

                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch

                    # Tell pytorch not to bother with constructing the compute graph during
                    # the forward pass, since this is only needed for backprop (training).
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions.
                        # token_type_ids is the same as the "segment ids", which
                        # differentiates sentence 1 and 2 in 2-sentence tasks.
                        # The documentation for this `model` function is here:
                        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                        # Get the "logits" output by the model. The "logits" are the output
                        # values prior to applying an activation function like the softmax.
                        (loss, logits) = bert_model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids, labels = label_ids)

                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.

                    norm_outputs = [softmax(out) for out in logits]

                    if not multilabel_mode:
                        y_pred = [np.argmax(arr) for arr in norm_outputs]
                    else:
                        y_pred = [[scipy.special.expit(nb) for nb in out] for out in logits]
                        y_pred_int = [[1 if nb > 0.5 else 0 for nb in out] for out in y_pred]


                    #print('Predictions ', y_pred)

                    val_pred.extend(y_pred)
                    val_pred_int.extend(y_pred_int)
                    val_labels.extend(label_ids)



                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(dataloader)

                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print()

                # average of precision recall, fscore, support

                #avg_precision = np.sum(total_eval_metrics[:,0])/len(validation_dataloader)
                if not multilabel_mode:
                    multi_confusion_matrix = multilabel_confusion_matrix(val_labels, val_pred)
                    conf_matrix = confusion_matrix(val_labels, val_pred)
                    print("Validation confusion matrix :")
                    print(multi_confusion_matrix)
                    print(conf_matrix)

                    metrics = precision_recall_fscore_support(val_labels, val_pred, average='weighted')
                    precision, recall, fscore, support = precision_recall_fscore_support(val_labels, val_pred, average=None)
                    print('Metrics : precision, recall, f_score, support ', metrics)
                    print('Metrics : precision, recall, f_score, support for each class ', precision, recall, fscore,
                          support)




                    # print("  Validation took: {:}".format(validation_time))


                else :

                    print('================== Error Analysis ====================================')
                    data = dataset.AnnotatedDataset()

                    def print_example(tuple_id, tuple_df, annotated_timexes):

                        print()

                        print(tuple_id)
                        example = tuple_df[tuple_df['tuple_id'] == tuple_id].to_dict('records')[0]
                        print('RI-Timex :')
                        print(annotated_timexes[(annotated_timexes['docname'] == example['docname']) & (
                                    annotated_timexes['id'] == example['Rtimexe'])])
                        print('Potential anchor :')
                        print(annotated_timexes[(annotated_timexes['docname'] == example['docname']) & (
                                    annotated_timexes['id'] == example['Ptimexe'])])
                        return None

                    print(dataloader.dataset)
                    print(set_features)
                    new_features = set_features.reset_index()
                    print(new_features)
                    for i in range(len(dataloader.dataset)):
                        if random.random() > 0.98:
                            if set(val_pred_int[i]) != set(val_labels[i]):
                                print('----------- Error -----------')
                                tuple_id = new_features['tuple_id'][i]
                                print_example(tuple_id, data.tuple_df, data.timexes)
                                print( 'Prediction :', val_pred_int[i])
                                print('Label :', val_labels[i])

                    print(val_labels[:10])
                    print(val_pred[:10])
                    print(val_pred_int[:10])

                    print('Label ranking average precision :')
                    print(sklearn.metrics.label_ranking_average_precision_score(val_labels, val_pred))

                    print()
                    print('Coverage error :')
                    print(sklearn.metrics.coverage_error(val_labels, val_pred))

                    print()
                    print('Ranking Loss :')
                    print(sklearn.metrics.label_ranking_loss(val_labels, val_pred))

                    for i in range(4):
                        y_true = [label[i] for label in val_labels]
                        y_pred = [pred[i] for pred in val_pred_int]

                        #multi_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
                        conf_matrix = confusion_matrix(y_true, y_pred)
                        print("Validation confusion matrix :")
                        #print(multi_confusion_matrix)
                        print(conf_matrix)

                        metrics = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
                        #precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
                        print('Metrics : precision, recall, f_score, support ', metrics)

                    print()
                    print('General Metrics : ')
                    flat_labels = [lb  for label in val_labels for lb in label]
                    flat_pred = [nb for pred in val_pred_int for nb in pred ]
                    metrics = precision_recall_fscore_support(flat_labels, flat_pred, average='binary', pos_label=1)
                    print(metrics)

                    val_metrics.append(metrics)

                return metrics

            if val_prop > 0:
                run_eval(validation_dataloader, val_set_features)
        if val_prop > 0:
            me = pd.DataFrame(val_metrics, columns = ['precision', 'recall', 'f-score'])
            me.to_excel('training_scores_' + str(datetime.now()) + '.xlsx')

        # ========================================
        #               Test
        # ========================================

        print('============================ TEST ==================================================================')


        test_input_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['input_id'].values],dtype=torch.long)
        test_input_mask = torch.tensor([ast.literal_eval(f) for f in test_set_features['input_mask'].values],dtype=torch.long)
        test_segment_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['segment_id'].values], dtype=torch.long)

        if not multilabel_mode:
            test_label_ids = torch.tensor([f for f in test_set_features['label_id'].values], dtype=torch.long)
        else:
            test_label_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['label'].values],
                                         dtype=torch.long)

        test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=train_batch_size)

        run_eval(test_dataloader, test_set_features)

    now = str(datetime.now())
    os.mkdir('models/bert_weights_' + now)
    bert_model.save_pretrained('models/bert_weights_' + now)
    return bert_model






