import pandas as pd
from expression import Expression
from TEER import teer_rules
from TEER.TEXer_English import temporal_processing, new_temporal_testing
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






'''

class TEERTemporalExtractor(TemporalExtractor):

    def __init__(self, pattern_path):
        self.pattern_path = pattern_path

    # contain training and testing modules. set parameters to call different modules.
    def extract_expressions(self, documents):

        expressions = []
        # read the input data

        for document in documents:
            texts = [document.text]
            if texts is None or len(texts) <= 0:
                print('no text available for processing --- interrupting')
                return
            result = new_temporal_testing(document, patterns =self.pattern_path)

            expressions += [[document.name, exp.get_text(), exp.start_char, exp.end_char] for exp in result]
            #expressions = [re.sub(r'(</TI>)', '', stg) for stg in expressions]

        data_result = pd.DataFrame(data = expressions, columns=['doc', 'text', 'start', 'end'])

        return data_result
        
'''


