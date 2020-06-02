import pandas as pd
import numpy as np
import agreementEvaluation
from sklearn.metrics import f1_score, cohen_kappa_score
import os



def evaluate_agreement(batch1, batch2, annotations):
    """
    This function evaluates the agreement between two batches of documents annotatted by different annotators
    :param batch1: the list of path for the xml files annotated by annotator 1
    :param batch2: the list of paths for the xml files annotated by annotator 2
    :return:
    """
    print()
    print()


    # FILTERING RELATIVE TIMEXES - AGREEMENT
    rel1, rel2 = [], [] #to contain the values for the "relative" attribute
    for i in range(len(batch1)):
        path1 = batch1[i]
        path2 = batch2[i]
        docname = os.path.basename(path1)
        print(docname)
        f_score, kappa, r1, r2 = agreementEvaluation.compare_filtering(path1, path2)
        rel1 += r1
        rel2 += r2
    print(rel1)
    print(rel2)
    print()
    print()
    print('=====================================================================')
    print()
    print('Global Filtering F1 score : ' + str(f1_score(rel1, rel2, pos_label='TRUE')))
    print('Global Filtering Cohen Kappa : ' + str(cohen_kappa_score(rel1, rel2)))

    # ANCHORLINKS AGREEMENT
    print()

    total_anchored, positives, equivalent, negatives, missing_links = 0, 0, 0, 0,0
    for i in range(len(batch1)):
        path1 = batch1[i]
        path2 = batch2[i]
        docname = os.path.basename(path1)
        print(docname)
        t,p, e, n, m = agreementEvaluation.anchorlink_agreement(path1, path2, annotations[annotations.docname == docname])
        positives += p
        total_anchored += t
        equivalent += e
        negatives += n
        missing_links += m

    print()
    print('=====================================================================')
    print()
    print('Global Link Strict Agreement : ' + str(positives * 100/total_anchored))
    print('Global Link Relaxed Agreement : ' + str((positives + equivalent) * 100/total_anchored))



## Add section_time to the original annotations

date_and_time = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')
sectimes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_sectimes_annotations.xlsx')
all_annotations = date_and_time.append(sectimes, ignore_index=True)

date_and_time = pd.read_excel('../Normalization/date_and_time.xlsx')
original_gs = pd.read_excel('../TimeDatasets/i2b2 Data/ritimexes_original_gs.xlsx')
original_gs.columns = original_gs.iloc[0]
original_gs = original_gs.drop([0])


docnames = date_and_time[date_and_time['test'] == False]['docname'].unique()
test_docname = date_and_time[date_and_time['test'] == True]['docname'].unique()
test_gs_docnames = original_gs['docname'].unique()


# first batches

batch1 = docnames[:10]
batch2 = docnames[10:20]
batch3 = docnames[20:30]

nicol = np.concatenate([batch1, batch2])
sumithra = np.concatenate([batch2, batch3])
louise = np.concatenate([batch1, batch3])

# Preparing batches for annotating the test document

batch_test_1 = test_gs_docnames[:10]
batch_test_2 = test_gs_docnames[10:20]
batch_test_3 = test_gs_docnames[20:30]

nicol_test = np.concatenate([batch_test_1, batch_test_2])
sumithra_test = np.concatenate([batch_test_2, batch_test_3])
louise_test = np.concatenate([batch_test_1, batch_test_3])


# common documents - annotator & batch

path_nicol_1 = ['Nicol/annotated_documents/' + docname for docname in batch1]
path_louise_1 = ['Louise/annotated_documents/' + docname for docname in batch1]

#evaluate_agreement(path_nicol_1, path_louise_1, all_annotations)


path_sumithra_2 = ['Sumithra/annotated_documents/' + docname for docname in batch2]
path_nicol_2 = ['Nicol/annotated_documents/' + docname for docname in batch2]
evaluate_agreement(path_sumithra_2, path_nicol_2, all_annotations)

path_louise_3 = ['Louise/annotated_documents/' + docname for docname in batch3]
path_sumithra_3 = ['Sumithra/annotated_documents/' + docname for docname in batch3]
#evaluate_agreement(path_louise_3, path_sumithra_3, all_annotations)

# test batches

print(batch_test_1)

path_nicol_21 = ['Nicol_2/annotated_documents/' + docname for docname in batch_test_1]
path_louise_21 = ['Louise_2/annotated_documents/' + docname for docname in batch_test_1]
evaluate_agreement(path_nicol_21, path_louise_21, all_annotations)