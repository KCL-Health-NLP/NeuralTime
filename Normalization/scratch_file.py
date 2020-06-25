
import random
import pandas as pd
import svm_anchoring
import map_custom_annotations
import data_analysis
import os
import embeddings
from sklearn.model_selection import train_test_split
import neural_models
import numpy as np
import map_custom_annotations

random.seed(42)

## saving the anchorlinks

filepaths = [ '../RI_Annotations/AnnotatedData/' + docname  for docname in os.listdir('../RI_Annotations/AnnotatedData') ]
anchorlinks, timexes = map_custom_annotations.annotated_files_to_dataframe(filepaths)

anchorlinks.to_excel('anchorlinks.xlsx')



"""
# training anchor classififation models with the original data

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')
data = ri_original_timexes

date_and_time = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')  # for now, original filtering
all_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')


mimicII_vectorizer = embeddings.MimicIIEmbeddingVectorizer()


y_anchor = data['Anchor'].to_numpy()
y_relation = data['Relation_to_anchor'].to_numpy()
y_relation = [ 'After' if r == 'A' else r for r in y_relation]

train_data, test_data, y_anchor_train, y_anchor_test, y_relation_train, y_relation_test = train_test_split(data, y_anchor, y_relation, test_size=0.15, random_state=0, stratify=y_anchor)



path= '../TimeDatasets/i2b2 Data/all_data/'

text_documents = {}
for filepath in data['docname'].unique():
    text_path = filepath + '.txt'
    text = open(path + text_path, 'r').read()
    text_documents[filepath] = text

mimicII_vectorizer.fit([text for text in text_documents.values()])

X_train, previous_ids_train = svm_anchoring.extract_features(train_data, text_documents, mimicII_vectorizer, date_and_time, all_timexes, normalize_numbers = False)
X_test, previous_ids_test = svm_anchoring.extract_features(test_data, text_documents, mimicII_vectorizer, date_and_time, all_timexes, normalize_numbers = False)

X_train = X_train.todense()
X_test = X_test.todense()

# transforming y

anchors = ['A', 'D', 'PA', 'P']
train_labels = np.array([[1 if anchor == anchors[i] else 0 for i in range(4)] for anchor in y_anchor_train])
test_labels = np.array([[1 if anchor == anchors[i] else 0 for i in range(4)] for anchor in y_anchor_test])
print(train_labels)
# training the model
network = neural_models.train_network(X_train, train_labels)

print(network.evaluate(X_test, test_labels))"""