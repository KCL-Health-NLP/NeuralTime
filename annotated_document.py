import os
from expression import Expression
import random
import pandas as pd

class AnnotatedDocument():

    def __init__(self, file_path, annotations, corpus):
        self.path = file_path
        file = open(file_path)
        text_document = file.read()
        self.text = text_document
        self.name, self.extension = os.path.splitext(os.path.basename(file_path))
        #self.tokenized_doc = spacy_model(text_document)
        self.annotations = annotations
        self.corpus = corpus
        self.extracted_expressions = []
        r = random.random()
        if r> 0.8:
            self.test = True
        else:
            self.test = False

    def annotations_to_spacy_standard(self):
        # this returns a dict

        for tuple in annotations[annotations['doc'] == self.name][['start', 'end', 'type', 'text']].itertuples(index = False, name = None):
            exp = Expression(self, None, None, typ= tuple[2], value=tuple[3], start_char=tuple[0], end_char=tuple[1])
            self.annotations += [exp]

    def annotations_to_df(self):
        ann = pd.DataFrame()
        for exp in self.annotations:
            ann = ann.append({'doc' : self.name, 'text' : exp.value, 'start' : exp.start_char, 'end' : exp.end_char, 'type' : exp.type}, ignore_index= True)
        return ann







    def print_expressions(self):
        for exp in self.temporal_expressions:
            print(self.tokenized_doc[exp.start: exp.end + 1])
            print(exp.type)
            print(exp.value)
            print()

