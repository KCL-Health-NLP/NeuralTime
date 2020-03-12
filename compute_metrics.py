# This module is used to test various approaches and compute their accuracy metrics on the annotated test documents
import pandas as pd
import statistics as stat


# quite simple for now
def compute_quality(annotations, pred_annotations):
    # the annotations are dataframes in the following format :
    #  COLUMNS : document, text, type, normalized_value
    # where "type" in one of : date, time, duration, frequency, age-related

    # first, we group the data by document
    doc_data = annotations.groupby('document_name')
    pred_doc_data = pred_annotations.groupby('document_name')

    n_docs = len(doc_data.first())

    ## first, we compare the recognition of temporal expressions : if the string extracted and the actual strings match

    precisions = []
    recalls = []

    tp = 0  # true positives
    for name in doc_data.groups:
        list_of_expr = doc_data.get_group(name)['expression'].values
        p = len(list_of_expr)
        pred_expr = pred_doc_data.get_group(name)['expression'].values
        a = len(pred_expr)
        for e in list_of_expr:
            if e in pred_expr:
                tp += 1

    precision = tp / len(pred_annotations)
    recall = tp / len(annotations)
    f1 = 2 * (precision * recall) / (precision + recall)

    print('Temporal Expression Recognition : ')
    print('precision ' + str(precision))
    print('recall ' + str(recall))
    print('f1 ' + str(f1))

    return None

t = [['doc_1', 'March 2020', None, None], ['doc_1', 'since 8 years', None, None], ['doc_2', 'September', None, None], ['doc_2', 'this morning', None, None]]
td = pd.DataFrame(data = t, columns=['document_name', 'expression', 'type', 'normalized_value'])

c = [['doc_1', 'March 2020', None, None], ['doc_1', 'since 8 years', None, None], ['doc_2', 'first I left', None, None], ['doc_2', 'this morning', None, None]]
cd = pd.DataFrame(data = c, columns=['document_name', 'expression', 'type', 'normalized_value'])

#compute_quality(td, cd)


