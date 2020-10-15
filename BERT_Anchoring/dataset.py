

import pandas as pd
import os
from pathlib import Path
from spacy.lang.en import English


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, tuple_id, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.tuple_id = tuple_id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label




class TupleInstance():

    """
    A RI-Timexe / Potential Anchor pair, used for classification.
    The difference between TupleInstance and InputExample is that TupleInstance  is the high-level representation
    of a training/test example, while InputExample contains actual input text.
    """

    def __init__(self, tuple_id, docname, rtimexe, ptimexe, is_anchor, before, after, equal, test):
        """Constructs a TupleInstance.
        Args:
            docname: string. the document name of the rtimexe
            rtimexe: string. the id of the relative timexe
            ptimexe: string. the id of the potential anchor date (another timexe from the same document)
            is_anchor: boolean. whether or not the rtimexe is anchored to the ptimexe.
            before, after, equal : booleans. if is_anchor, the value is true for the correct anchor relation
            test: boolean. wether the instance is part of the test set.
        """
        self.tuple_id = tuple_id
        self.docname = docname
        self.rtimexe = rtimexe
        self.ptimexe = ptimexe
        self.is_anchor = is_anchor
        self.before = before
        self.after = after
        self.equal = equal
        self.test = test




class AnnotatedDataset():

    def __init__(self):
        here = Path(__file__).parent
        path = here / 'DataTables'
        self.timexes = pd.read_excel((path / 'annotated_timexes.xlsx').as_posix())
        self.anchorlinks = pd.read_excel((path /'anchorlinks.xlsx').as_posix())
        self.tuple_df = pd.read_excel((path /'tuple_df.xlsx').as_posix())
        self.train_inputs = pd.read_excel((path / 'train_inputs.xlsx').as_posix())
        self.inference_tuple_df = pd.read_excel((path / 'inference_tuple_df.xlsx').as_posix())
        self.inference_train_inputs = pd.read_excel((path / 'inference_train_inputs.xlsx').as_posix())
        self.test_inputs = pd.read_excel((path / 'test_inputs.xlsx').as_posix())
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
            anchor = self.timexes[(self.timexes['docname'] == docname) & (self.timexes['id'] == anchor_link['toID'])].to_dict('records')[0]
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
        return [id_ for id_ in doc_timexes]

    def get_doc_text(self, docname):
        here = Path(__file__).parent
        path = here.parent.parent / 'TimeDatasets/i2b2 Data/all_data' / (docname + '.txt')
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
        tuple_instances = []

        for timexe in r_timexes.to_dict('records'):
            docname = timexe['docname']
            id1 = timexe['id']
            test = timexe['test']
            val = timexe['val']
            """print()
            print()
            print('Current timexe ', id1, val)
            print(docname)"""
            potential_anchors = [ self.get_timexe(docname, id2) for id2 in [ pa_id for pa_id in self.get_timexes_ids(docname) if pa_id != id1]]
            anchor = self.get_anchor(docname, id1)
            """if anchor is not None:
                #print('Anchor ', anchor['val'])
                #print()"""
            for pa in potential_anchors:
                if anchor is not None:
                    is_anchor = anchor['id'] == pa['id']
                    if inference:
                        #print(pa['annotated_relative'], pa['val'])
                        # adding the absolute timexes whose value is the same as the anchor
                        if not pa['annotated_relative'] and anchor['val'] == pa['val']:
                            print("Inference")
                            is_anchor = True
                    if is_anchor:
                        relation = anchor['anchor_relation']
                    else:
                        relation = None
                else :
                    is_anchor = False
                    relation = None
                tuple_id = docname + '_' + id1 + '_' + pa['id']
                tuple_list += [[tuple_id, docname, id1, pa['id'], is_anchor, relation == 'BEFORE', relation == 'EQUAL', relation == 'AFTER', test]]
                tuple_instances += [TupleInstance(tuple_id, docname, id1, pa['id'], is_anchor, relation == 'BEFORE', relation == 'EQUAL', relation == 'AFTER', test )]

        self.tuple_df = pd.DataFrame(tuple_list, columns=['tuple_id','docname', 'Rtimexe', 'Ptimexe', 'is_anchor', 'Before', 'Equal', 'After', 'test'])

        if out_path is not None:
            self.tuple_df.to_excel(out_path)

        return tuple_instances



    def generate_inputs(self, type = 'train', out_path = None, inference = False):

        """
        The goal is to output text sequences that have been correctly processed for Bert training and testing.
        parameters of the InputExamples
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.


        :return: a list of InputExamples
        """

        # Note : the spacy tokenizer is only used to select the section of text to be passed as inputs.
        # Bert uses its own tokenizer

        # for each example in the dataset , generate an InputExample

        examples = []
        inputs = []

        if inference:
            df = self.inference_tuple_df
        else:
            df = self.tuple_df

        if type == 'test':
            tuple_data = df[df['test']]
        else:
            tuple_data = df[df['test'] == False]


        print('analysis : ')
        print(len(tuple_data[tuple_data['is_anchor'] == True]), ' true anchor relations')


        for example in tuple_data.to_dict('records'):

            docname = example['docname']
            id1 = example['Rtimexe']
            id2 = example['Ptimexe']
            tuple_id = example['tuple_id']

            timexe2 = self.get_timexe(docname, id2)
            # we prepare the two input sequences
            # by putting together window of tokens around the timexes and tags to specify the position of the timexe
            timexe_text1, window_left1, window_right1 = self.get_window(docname, id1, 200)
            # ts and te are the tag for the timexe to be anchored
            text_a = window_left1 + ' re ' + timexe_text1 + ' te ' + window_right1

            timexe_text2, window_left2, window_right2 = self.get_window(docname, id2, 200)
            # differentiation for relative/absolute in the tags
            if timexe2['annotated_relative'] :
                text_b = window_left2 + ' es ' + timexe_text2  + ' et ' + window_right2
            else:
                text_b = window_left2 + ' es ' + timexe_text2 + ' et ' + window_right2

            label = [int(example['is_anchor']), int(example['Before']), int(example['Equal']), int(example['After'])]

            examples.append(InputExample(docname + '_' + id1, text_a, text_b, label))
            inputs.append([tuple_id, text_a, text_b, label ])

        input_dataset = pd.DataFrame(inputs, columns = ['tuple_id', 'text_a', 'text_b', 'label'])
        if out_path is not None:
            input_dataset.to_excel(out_path)
        return examples

    def get_examples(self, inference = False,  type = 'train', input_data = None):
        examples = []

        if input_data is not None:
            examples = [InputExample(id_, text_a, text_b, label) for id_, text_a, text_b, label in
                        zip(input_data.tuple_id, input_data.text_a, input_data.text_b,
                            input_data.label)]
            return examples



        if inference :
            train_example_df = self.inference_train_inputs
        else:
            train_example_df = self.train_inputs
        if type == 'train':

            examples = [InputExample(id_, text_a, text_b, label) for id_, text_a, text_b, label in zip(train_example_df.tuple_id, train_example_df.text_a, train_example_df.text_b, train_example_df.label)]
        elif type == 'test':
            examples = [InputExample(id_, text_a, text_b, label) for id_, text_a, text_b, label in zip(self.test_inputs.tuple_id, self.test_inputs.text_a, self.test_inputs.text_b,
                            self.test_inputs.label)]
        return examples

    def print_stats(self):
        print( '==================== Annotated Dataset ============================================================== ')
        print()
        print('Inference Anchor Dates :', self.inference)
        pass


















