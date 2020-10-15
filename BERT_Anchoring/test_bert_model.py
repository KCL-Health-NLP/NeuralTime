import pandas as pd
import sklearn
import transformers
import os
import ast
import random
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix
from bert_for_multilabel_classification import BertForMultiLabelSequenceClassification
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from bert_transformer import softmax
import scipy
import numpy as np
import torch
import dataset


"""
This script performs test and error analysis of a given bert model
"""

model_path = 'models/bert_weights_2020-08-13 15:32:08.376929'    # the path to the bert model's weight directory

# data loading

data= dataset.AnnotatedDataset()
test_features = pd.read_excel('DataTables/test_features.xlsx')
inf_test_features = pd.read_excel('DataTables/inference_test_features.xlsx')


# Prepare model

last_bert_model = BertForMultiLabelSequenceClassification.from_pretrained(model_path)
#bert_model.classifier = torch.nn.Linear(768, 4)
#bert_model.num_labels = 4




def print_example(tuple_id, tuple_df, annotated_timexes):

    print()

    print(tuple_id)
    example = tuple_df[tuple_df['tuple_id'] == tuple_id].to_dict('records')[0]
    print('RI-Timex :')
    print(annotated_timexes[(annotated_timexes['docname'] == example['docname']) & (annotated_timexes['id'] == example['Rtimexe'])])
    print('Potential anchor :')
    print(annotated_timexes[(annotated_timexes['docname'] == example['docname']) & (annotated_timexes['id'] == example['Ptimexe'])])
    return None


def eval_and_error_analysis(test_set_features, data,  bert_model, multilabel_mode = True, batch_size = 5):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    bert_model.to(device)

    print("")
    print("Running Test...")

    test_input_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['input_id'].values], dtype=torch.long)
    test_input_mask = torch.tensor([ast.literal_eval(f) for f in test_set_features['input_mask'].values],
                                   dtype=torch.long)
    test_segment_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['segment_id'].values],
                                    dtype=torch.long)

    if not multilabel_mode:
        test_label_ids = torch.tensor([f for f in test_set_features['label_id'].values], dtype=torch.long)
    else:
        test_label_ids = torch.tensor([ast.literal_eval(f) for f in test_set_features['label'].values],
                                      dtype=torch.long)

    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
    test_sampler = RandomSampler(test_data)
    dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


    # t0 = time.time()

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
            try:
                (loss, logits) = bert_model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                                        labels=label_ids)
            except Exception as e:
                print(e)
                print(input_ids)
                print(label_ids)

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

        # print('Predictions ', y_pred)







        val_pred.extend(y_pred)
        val_pred_int.extend(y_pred_int)
        val_labels.extend(label_ids)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(dataloader)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print()

    # average of precision recall, fscore, support

    # avg_precision = np.sum(total_eval_metrics[:,0])/len(validation_dataloader)
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


    else:

        print('================== Error Analysis ====================================')

        for i in range(len(test_set_features)):
            if random.random() > 0.7:
                if set(val_pred_int[i]) != set(val_labels[i]) :
                    print('----------- Error -----------')
                    tuple_id = test_features['tuple_id'][i]
                    print_example(tuple_id, data.tuple_df, data.timexes)
                    print('prediction ', val_pred_int[i])
                    print('label ', val_labels[i])


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

            # multi_confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
            conf_matrix = confusion_matrix(y_true, y_pred)
            print("Validation confusion matrix :")
            # print(multi_confusion_matrix)
            print(conf_matrix)

            metrics = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
            # precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=None)
            print('Metrics : precision, recall, f_score, support ', metrics)

        print()
        print('General Metrics : ')
        flat_labels = [lb for label in val_labels for lb in label]
        flat_pred = [nb for pred in val_pred_int for nb in pred]
        metrics = precision_recall_fscore_support(flat_labels, flat_pred, average='binary', pos_label=1)
        print(metrics)

    return metrics

#eval_and_error_analysis(test_features, data, last_bert_model)
eval_and_error_analysis(inf_test_features, data, last_bert_model)