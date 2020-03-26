import pandas as pd
import re
from spacy.gold import GoldParse


class SpacyTemporalExtractor():

    def __init__(self, spacy_model):
        self.model = spacy_model

    def extract_expressions(self, documents):

        documents['tokens'] = documents ['text'].apply(lambda x : self.model(x))

        exprs = [ [(doc, text[e.start_char:e.end_char], e.start_char, e.end_char, e.label_) for e in tokens.ents if e.label_ in ['DATE', 'TIME', 'FREQUENCY', 'DURATION', 'AGE_RELATED']] for [doc, text, tokens] in documents[['docname', 'text', 'tokens']].to_numpy()]
        exprs = [e for exp in exprs for e in exp]
        expressions = pd.DataFrame(exprs,  columns=['doc', 'text', 'start', 'end', 'type'])


        return expressions







