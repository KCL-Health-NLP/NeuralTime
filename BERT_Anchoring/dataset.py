

import pandas as pd
import os
from pathlib import Path
from spacy.lang.en import English


class AnnotatedDataset():

    def __init__(self):
        here = Path(__file__).parent
        path = here / 'DataTables'
        self.timexes = pd.read_excel((path / 'annotated_timexes.xlsx').as_posix())
        self.anchorlinks = pd.read_excel((path /'anchorlinks.xlsx').as_posix())
        self.tuple_df = pd.read_excel((path /'tuple_format.xlsx').as_posix())
        self.nlp = English()

    def get_timexe(self, docname, id):

        try:
            timexe = self.timexes[(self.timexes['docname'] == docname) & (self.timexes['id'] == id)].to_dict('records')[0]
            return timexe
        except Exception as e:
            print('No timexe found')
            return None

    def get_anchor(self, docname, id):
        """
        Gets the anchor for a specified relative timexe
        :param docname:
        :param id:
        :return: a dict object with the anchor timexe attributes + an anchor_relation attribute
        """
        try:
            anchor_link = self.anchorlinks[(self.anchorlinks['docname'] == docname) & (self.anchorlinks['fromID'] == id)].to_dict('records')[0]
            anchor = self.timexes[self.timexes['id'] == anchor_link['toID']].to_dict('records')[0]
            anchor['anchor_relation'] = anchor_link['relation']
            return anchor
        except Exception as e:
            print('No anchor date was found')
            return None

    def is_anchor(self, docname, id1, id2):

        """
        Tests wether the first timexe is anchored to the second and specifies the relation
        :param docname: document name
        :param id1: id of the first timexe
        :param id2: id of the potential anchor
        :return: anchore, anchor_relation or False, None
        """
        anchor  = self.get_anchor(docname, id1)
        print(anchor)
        if anchor is not None:
            if id2 == anchor['id']:
                return True, anchor['anchor_relation']
        return False, None

    def get_timexes_ids(self, docname):
        """
        returns a list of the timexes ids for the given document
        """
        doc_timexes = self.timexes[self.timexes['docname'] == docname]['id']
        print(doc_timexes)
        print([id_ for id_ in doc_timexes])
        return [id_ for id_ in doc_timexes]

    def get_doc_text(self, docname):
        here = Path(__file__).parent
        path = here.parent / 'TimeDatasets/i2b2 Data/all_data' / (docname + '.txt')
        text = open(path.as_posix()).read()
        return text

    def get_window(self, docname, id, size):
        """
        return a widow of approximately the given size, with tokens on each side of the id
        :param docname: document name
        :param id: timexe id
        :param size: number of tokens to select from each side
        :return: three string, timexe text, left and right side of the window
        """
        timexe = self.get_timexe(docname, id)
        start = timexe['start']
        end = timexe['end']

        text = self.get_doc_text(docname)

        timexe_text= text[timexe['start'] -1 : timexe['end'] -1]
        tokens_left = self.nlp(text[0:start -1])
        tokens_right = self.nlp(text[end -1:])
        window_left = ''
        window_right = ''
        if len(tokens_left) > 0:
            window_left = tokens_left[ - min(len(tokens_left) -1, size//2):].text
        if len(tokens_right) > 0:
            window_right = tokens_right[:min(len(tokens_right) - 1, size//2)].text

        return timexe_text, window_left, window_right


    def convert_to_tuples(self, out_path = None, inference = False):
        """
        This function converts the dataset to a a list of (RTimexe, Timexe) pairs where the potential anchoring and
        anchor relation is specified.
        :param outpath : the path to save the dataframe
        :param inference : whether or not to consider infered anchor dates as valid anchor dates
        :return: a dataframe with the following columns :
                    - docname
                    - RTimexe - the relative time expression
                    - PTimexe - the potential anchor date
                    - Anchor - a Boolean specifying whethter the Rtimexe is anchored to the PTimexe
                    - Before - Boolean : anchor relation
                    - Equal - Boolean : anchor relation
                    - After - Boolean : anchor relation
                    - test - wether of not the example is part of the test set

        """

        r_timexes = self.timexes[self.timexes['annotated_relative']]

        tuple_list = []
        for timexe in r_timexes.to_dict('records'):
            docname = timexe['docname']
            id1 = timexe['id']
            test = timexe['test']
            val = timexe['val']
            potential_anchors = [ self.get_timexe(docname, id2) for id2 in [ pa_id for pa_id in self.get_timexes_ids(docname) if pa_id != id1]]
            anchor = self.get_anchor(docname, id1)
            for pa in potential_anchors:
                if anchor is not None:
                    is_anchor = anchor['id'] == pa['id']
                    if is_anchor:
                        relation = anchor['anchor_relation']
                    else:
                        relation = None
                else :
                    is_anchor = False
                    relation = None
                tuple_list += [[docname, id1, pa['id'], is_anchor, relation == 'BEFORE', relation == 'EQUAL', relation == 'AFTER', test]]

        self.tuple_df = pd.DataFrame(tuple_list, columns=['docname', 'Rtimexe', 'Ptimexe', 'is_anchor', 'Before', 'Equal', 'After', 'test'])

        if out_path is not None:
            self.tuple_df.to_excel(out_path)

        return self.tuple_df
















