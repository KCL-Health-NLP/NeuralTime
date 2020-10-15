import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
import pandas as pd
import transformers
import numpy as np
from dataset import AnnotatedDataset
from tqdm import trange, tqdm
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
import ast


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def map_label(label_string):
    """
    Utility function to map the labels from input dataframe into a format suited to Bert classifier
    :return:
    """
    # cast to list
    label = ast.literal_eval(label_string)

    # convert to a multiclassification output
    label_list = [int(not bool(label[0])), label[1], label[2], label[3]]

    return label_list



class BertDataset(Dataset):

    def __init__(self, annotatedData, input_path):
        """
        Args:
           annotatedData : an instance of AnnotatedDataset
        """
        self.data = annotatedData
        self.inputs = pd.read_excel(input_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        input = self.inputs[self.inputs['id'] == idx].to_dict('records')[0]
        input_example = InputExample(input['id'], input['text_a'], input['text_b'], map_label(input['label']))
        return input_example


    def generate_input(self, type = 'train', out_file = None):

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

        if type == 'test':
            tuple_data = self.data.tuple_df[self.data.tuple_df['test']]
        else:
            tuple_data = self.data.tuple_df[self.data.tuple_df['test'] == False]

        for example in tuple_data.to_dict('records'):

            print(example)
            docname = example['docname']
            id1 = example['Rtimexe']
            id2 = example['Ptimexe']

            timexe2 = self.data.get_timexe(docname, id2)
            # we prepare the two input sequences
            # by putting together window of tokens around the timexes and tags to specify the position of the timexe
            timexe_text1, window_left1, window_right1 = self.data.get_window(docname, id1, 200)
            # ts and te are the tag for the timexe to be anchored
            text_a = window_left1 + ' re ' + timexe_text1 + ' te ' + window_right1

            timexe_text2, window_left2, window_right2 = self.data.get_window(docname, id2, 200)
            # differentiation for relative/absolute in the tags
            if timexe2['annotated_relative'] :
                text_b = window_left2 + ' es ' + timexe_text2  + ' et ' + window_right2
            else:
                text_b = window_left2 + ' es ' + timexe_text2 + ' et ' + window_right2

            label = [int(example['is_anchor']), int(example['Before']), int(example['Equal']), int(example['After'])]

            examples.append(InputExample(docname + '_' + id1, text_a, text_b, label))
            inputs.append([docname + '_' + id1, text_a, text_b, label ])

            print()
        input_dataset = pd.DataFrame(inputs, columns = ['id', 'text_a', 'text_b', 'label'])
        if out_file is not None:
            input_dataset.to_excel(out_file)
        return examples


    def get_examples(self, type = 'train'):
        examples = []
        if type == 'train':
            df = pd.read_excel('DataTables/train_inputs.xlsx')
            # we modify the label to adapt the problem to a multiclass classification (4 cases : Not Anchored, Before, Equal, After)
            examples = [InputExample(id_, text_a, text_b, map_label(label)) for id_, text_a, text_b, label in zip(df.id, df.text_a, df.text_b, df.label)]
        return examples

    @staticmethod
    def get_labels():
        return ['NA', 'Before', 'Equal', 'After']


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    max_len = 0
    for  idx, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            seq_len = len(tokens_a) + len(tokens_b)

            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            seq_len = len(tokens_a)
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if seq_len > max_len:
            max_len = seq_len
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        print(example.label)
        label_id = example.label.index(1)
        if idx < 3:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    print('Max Sequence Length: %d' % max_len)

    return features


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)



def clinical_bert(
        do_train, max_seq_length, learning_rate, num_train_epochs, gradient_accumulation_steps, train_batch_size,
        warmup_proportion ):

    # initializing the models
    bert_tokenizer = transformers.AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    bert_model = transformers.BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # preparing train and test data

    ann_data = AnnotatedDataset()
    bert_data = BertDataset(ann_data, 'DataTables/train_inputs.xlsx')

    train_examples = None
    num_train_optimization_steps = None

    # getting train examples
    train_examples = bert_data.get_examples('train')
    label_list = bert_data.get_labels()
    num_train_optimization_steps = int(len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

    # Prepare model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    n_gpu = torch.cuda.device_count()
    bert_model.to(device)

    # Prepare optimizer
    param_optimizer = list(bert_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=learning_rate,)


    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if do_train:
        train_features = convert_examples_to_features( train_examples, label_list, max_seq_length, bert_tokenizer)
        print("***** Running training *****")
        print("  Num examples = %d", len(train_examples))
        print("  Batch size = %d", train_batch_size)
        print("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= train_batch_size)

        bert_model.train()
        for _ in trange(int(num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                print(input_ids.shape, input_mask.shape, segment_ids.shape, label_ids.shape)
                loss = bert_model(input_ids = input_ids, attention_mask = input_mask, token_type_ids = segment_ids, labels = label_ids) # label_ids not used ?

                print(loss)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1







