
import spacy
import random
nlp = spacy.load('en_core_web_sm')
from corpus import Corpus
from temporal_extractor import SpacyTemporalExtractor
from ehost_agreement_general import batch_agreement
import pandas as pd
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from utilities import merge_intervals
import re
from sklearn.model_selection import KFold, StratifiedKFold


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return cleaned_data



def test_model(test_docs, nlp, spacy_type = False):
    print(" ============= TESTING MODEL ===========================")

    # the annotations to be tested by batch_agreement are dataframes in the form ['doc', 'start', 'end', 'text', 'attribute1', 'attribute2', ..]
    test_annotations = []
    for [docname, annotations, text] in test_docs[['docname', 'annotations', 'text']].to_numpy():
        for ann in annotations:
            test_annotations += [(docname, ann[0], ann[1], text[ann[0]: ann[1]], ann[2])]

    test_annotations = pd.DataFrame(test_annotations, columns=['doc', 'start', 'end', 'text', 'type'])

    spacy_extractor = SpacyTemporalExtractor(nlp)
    spacy_annotations = spacy_extractor.extract_expressions(test_docs)

    tp_g, fp_g, fn_g, p, r, f, pt, rt, ft, type_dict = batch_agreement(test_docs['docname'].to_numpy(), test_annotations, spacy_annotations, attrs_to_check=['type'])

    # test by type

    return tp_g, fp_g, fn_g, p, r, f, pt, rt, ft, type_dict



def train_model_cross_validation(model, train_docs, test_docs, nb_iter, output_dir, spacy_type = True, other_annotations = False):

    print(" ============= TRAINING MODEL ===========================")

    # tuple conversion (the tuple type is lost when dataframe -> excel -> dataframe)

    #docs['annotations'] = [[tuple(ann) for ann in annotations] for annotations in docs['annotations'].to_numpy()]

    # cross validation :

    models = []
    all_scores = []

    kf = KFold(n_splits=5)
    c = 0
    for train_index, val_index in kf.split(train_docs):

        train_data = train_docs.iloc[train_index, :]
        val_data = train_docs.iloc[val_index, :]

        # spacy_format
        TRAIN_DATA = [(text, {'entities': entities}) for [text, entities] in train_data[['text', 'annotations']].to_numpy()]

        # trim entities : leading whitespace make the model bug
        TRAIN_DATA = trim_entity_spans(TRAIN_DATA)

        # loading of the model
        nlp = model

        optimizer = nlp.begin_training()

        # get names of other pipes to disable them during training
        pipe_exceptions = ["ner" ]  #"trf_wordpiecer", "trf_tok2vec"
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

        scores = []

        # training
        with nlp.disable_pipes(*other_pipes):  # only train NER

            if not spacy_type : # add the other labels
                ner = nlp.get_pipe("ner")
                ner.add_label('AGE_RELATED')
                ner.add_label('DURATION')
                ner.add_label('FREQUENCY')
                ner.add_label('OTHER')

            for i in range(nb_iter):

                print('Iteration ', i)
                print()
                losses = {}
                random.shuffle(TRAIN_DATA) # ??

                path = ''
                if spacy_type:
                    path = 'spacy_model_' + str(c) + 'fold'
                else:
                    path = 'all_types_model_' + str(c) + 'fold'

                batches = minibatch(TRAIN_DATA, size=1)  #compounding(4.0, 20.0, 1.001)

                for batch in batches:
                    texts, annotations = zip(*batch)
                    try:
                        nlp.update(texts, annotations, sgd = optimizer,  drop=0.5, losses = losses)
                        print("Losses", losses)
                    except Exception as e:
                        print(e)
                        #print(text)

                scores += [test_model(test_docs, nlp)]
                print()
                print()

            # test the trained model
            test_model(val_data, nlp)

            df_scores = pd.DataFrame(scores, columns = ['span_precision', 'span_recall', 'span_f1', 'type_precision', 'type_recall', 'type_f1'])
            df_scores.to_excel(path + '.xlsx')

            models += [nlp]
            all_scores += [scores]
            # save model to output directory
            if output_dir is not None:
                nlp.to_disk(output_dir + '/' + path)
                print("Saved model to", output_dir + '/' + path)

        c += 1

    return models, all_scores


def train_model(model, train_docs, test_docs, nb_iter, output_dir, spacy_type = True, other_annotations = False):

    print(" ============= TRAINING MODEL ===========================")

    # tuple conversion (the tuple type is lost when dataframe -> excel -> dataframe)

    #docs['annotations'] = [[tuple(ann) for ann in annotations] for annotations in docs['annotations'].to_numpy()]

    # cross validation :

    models = []
    all_scores = []


    train_data = train_docs

    # spacy_format
    TRAIN_DATA = [(text, {'entities': entities}) for [text, entities] in train_data[['text', 'annotations']].to_numpy()]

    # trim entities : leading whitespace make the model bug
    TRAIN_DATA = trim_entity_spans(TRAIN_DATA)

    # loading of the model
    nlp = model

    optimizer = nlp.begin_training()

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner" ]  #"trf_wordpiecer", "trf_tok2vec"
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    scores = []

    # training
    with nlp.disable_pipes(*other_pipes):  # only train NER

        if not spacy_type : # add the other labels
            ner = nlp.get_pipe("ner")
            ner.add_label('AGE_RELATED')
            ner.add_label('DURATION')
            ner.add_label('FREQUENCY')
            ner.add_label('OTHER')

        for i in range(nb_iter):

            print('Iteration ', i)
            print()
            losses = {}
            random.shuffle(TRAIN_DATA) # ??

            path = ''
            if spacy_type:
                path = 'spacy_model_scores_on_all_data'
            else:
                path = 'all_types_model_on_all_data'

            batches = minibatch(TRAIN_DATA, size=1)  #compounding(4.0, 20.0, 1.001)

            for batch in batches:
                texts, annotations = zip(*batch)
                try:
                    nlp.update(texts, annotations, sgd = optimizer,  drop=0.5, losses = losses)
                    print("Losses", losses)
                except Exception as e:
                    print(e)
                    #print(text)

            tp_g, fp_g, fn_g, p, r, f, pt, rt, ft, type_dict = test_model(test_docs, nlp)
            scores += [(p, r, f, pt, rt, ft)]
            print()
            print()

            # test the trained model

            df_scores = pd.DataFrame(scores, columns = ['span_precision', 'span_recall', 'span_f1', 'type_precision', 'type_recall', 'type_f1'])
            df_scores.to_excel(path + '.xlsx')

            # save model to output directory
            if output_dir is not None:
                nlp.to_disk(output_dir + '/' + path)
                print("Saved model to", output_dir + '/' + path)

    return models, all_scores






'''
def extract_training_data( documents, spacy_model, spacy_type,  other_entities = False):
    # This function is useful to solve the "catastrophic forgetting" problem.
    # We add the other annotations to the data so the model does not forget how to generalise

    TRAIN_DATA = []

    for ann_doc in documents:
        print(ann_doc.name)
        doc = spacy_model(ann_doc.text)
        tags = [w.tag_ for w in doc]
        heads = [w.head.i for w in doc]
        deps = [w.dep_ for w in doc]

        # first the temporal annotations
        temp_entities = annotations_to_spacy_standard(ann_doc, spacy_type)

        # we remove the overlapping annotations
        print('before removing')
        print(temp_entities)
        temp_entities = [e for e in temp_entities if temp_entities.count(e) == 1]
        temp_entities = temp_entities_overlap(temp_entities)
        print('after removing')
        print(temp_entities)
        entities = temp_entities

        if other_entities:

            #the other annotations
            other_entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents if
                        (e.label_ != 'DATE' and e.label_ != 'TIME')]
            print(other_entities)

            # check that no entities overlap :
            other_entities = entities_overlap(temp_entities, other_entities)

            # we concatenate all the annotations
            entities += other_entities
        entities.sort()
        print(entities)
        TRAIN_DATA += [(doc.text, {'entities': entities})]


    return TRAIN_DATA

        #revision_data.append((doc, GoldParse(doc, tags=tags, heads=heads, deps=deps, entities=entities)))
'''

#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""



'''# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)'''
