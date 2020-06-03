import pandas as pd
import numpy as np
import agreementEvaluation
import convert_xml
from sklearn.metrics import f1_score, cohen_kappa_score
import os



date_and_time = pd.read_excel('../Normalization/date_and_time.xlsx')

## Add section_time to the original annotations

sectimes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_sectimes_annotations.xlsx')
all_annotations = date_and_time.append(sectimes, ignore_index=True)




# Original annotations
original_gs = pd.read_excel('../TimeDatasets/i2b2 Data/ritimexes_original_gs.xlsx')
original_gs.columns = original_gs.iloc[0]
original_gs = original_gs.drop([0])
print(original_gs)


docnames = date_and_time[date_and_time['test'] == False]['docname'].unique()
test_docname = date_and_time[date_and_time['test'] == True]['docname'].unique()
test_gs_docnames = original_gs['docname'].unique()


# ==================================== Preparing batches ===============================================================

# First batches

batch1 = docnames[:10]
batch2 = docnames[10:20]
batch3 = docnames[20:30]

nicol = np.concatenate([batch1, batch2])
sumithra = np.concatenate([batch2, batch3])
louise = np.concatenate([batch1, batch3])


# second batches


# Preparing batches for annotating the test document

batch_test_1 = test_gs_docnames[:10]
batch_test_2 = test_gs_docnames[10:20]
batch_test_3 = test_gs_docnames[20:30]

nicol_test = np.concatenate([batch_test_1, batch_test_2])
sumithra_test = np.concatenate([batch_test_2, batch_test_3])
louise_test = np.concatenate([batch_test_1, batch_test_3])

## final batches

train_batch_nicol = docnames[30:110]  # 80 train documents
train_batch_louise = docnames[110:]


remaining_test_docs = [doc for doc in test_docname if doc not in np.concatenate([batch_test_1, batch_test_2, batch_test_3])]
print(len(remaining_test_docs))

nicol_test_batch = remaining_test_docs[:50]
louise_test_batch = remaining_test_docs[45:]



convert_xml.prepare_batch(train_batch_nicol, 'Nicol_Train', local_path='../TimeDatasets/i2b2 Data/Train-2012-07-15')
convert_xml.prepare_batch(train_batch_louise, 'Louise_Train', local_path='../TimeDatasets/i2b2 Data/Train-2012-07-15')
convert_xml.prepare_batch(louise_test_batch, 'Louise_Test', local_path='../TimeDatasets/i2b2 Data/Test_data/merged_xml' )
convert_xml.prepare_batch(nicol_test_batch, 'Nicol_Test', local_path='../TimeDatasets/i2b2 Data/Test_data/merged_xml' )



# ================================== Evaluating Agreement ==============================================================


# common documents - annotator & batch

path_nicol_1 = ['Nicol/annotated_documents/' + docname for docname in batch1]
path_louise_1 = ['Louise/annotated_documents/' + docname for docname in batch1]

#evaluate_agreement(path_nicol_1, path_louise_1, all_annotations)


path_sumithra_2 = ['Sumithra/annotated_documents/' + docname for docname in batch2]
path_nicol_2 = ['Nicol/annotated_documents/' + docname for docname in batch2]
agreementEvaluation.evaluate_agreement(path_sumithra_2, path_nicol_2, all_annotations)

path_louise_3 = ['Louise/annotated_documents/' + docname for docname in batch3]
path_sumithra_3 = ['Sumithra/annotated_documents/' + docname for docname in batch3]
#evaluate_agreement(path_louise_3, path_sumithra_3, all_annotations)

# test batches

print(batch_test_1)

path_nicol_21 = ['Nicol_2/annotated_documents/' + docname for docname in batch_test_1]
path_louise_21 = ['Louise_2/annotated_documents/' + docname for docname in batch_test_1]
agreementEvaluation.evaluate_agreement(path_nicol_21, path_louise_21, all_annotations)