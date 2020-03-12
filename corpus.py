import pandas as pd
import spacy
import numpy as np
from spacy.gold import GoldParse
import pandas as pd
from annotated_document import AnnotatedDocument

class Corpus():

    def __init__(self, name, annotations):
        self.name = name
        self.annotations = annotations
        self.documents = []


    def populate(self, file_path_list):
        # filepath = a list of tuple name, path representing the documents of the corpus
        for name, path in file_path_list:
            self.documents += [AnnotatedDocument(file_path = path, annotations = self.annotations[self.annotations['doc'] == name], corpus = self.name)]





    def annotations_to_extended_standard(self, file_path, corpus):
        data = pd.read_excel(file_path)

        print(len(data))
        data_list = []

        for index, row in data.iterrows():
            if row[1] == 'TP' :
                data_list.append([row[0].strip(), row[2], row[4],row[8], row[9], corpus, '0']) # both annotators
            if row[1] == 'FN':
                data_list.append([row[0].strip(), row[2], row[4], row[8], row[9], corpus, '1'])  # according to first annotator
            elif row[1] == 'FP' :
                data_list.append( [row[0].strip(),  row[3], row[5],  row[10],  row[11],  corpus, '2'])  # second annotator

        s_data = pd.DataFrame(data_list, columns = ['doc', 'text', 'type', 'start', 'end', 'corpus', 'annotator'])
        print(len(s_data))
        return s_data

    def annotations_to_spacy_standard(self, doc, spacy_type = True):
        # docs : annotated documents
        # the annotations are converted to spacy NER annotations, to be used in training the statistical model

        # if spacy_type, the types are converted to spacy temporal types : either DATE, or TIME

        entities = []
        for exp in doc.annotations:
            if spacy_type:
                if exp.type == 'time':
                    tp =  'TIME'
                else :
                    tp = 'DATE'
            else:
                tp = exp.type.upper()
            entities += [(exp.start_char, exp.end_char, tp)]
        return entities







