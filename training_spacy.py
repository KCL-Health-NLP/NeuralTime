
import spacy
import random
nlp = spacy.load('en_core_web_sm')
from temporal_extractor import SpacyTemporalExtractor
from compute_metrics import batch_agreement
import pandas as pd
import random
import spacy
from spacy.util import minibatch, compounding
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

    """
    Tests a model
    :param test_docs: the documents to be tested, in a dataframe format with columns 'docname', 'annotations', and 'text'
    :param nlp: the spacy model to test
    :param spacy_type: wether spacy types or the standard types are used
    :return: the result of batch_agreement (global metrics function) on the annotated test set
    """
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





def train_model_cross_validation(model, train_docs, test_docs, nb_iter, output_dir, spacy_type = True, nb_folds = 5):

    """
    Trains a model using a cross validation technique
    :param model: the spacy model to retrain
    :param train_docs: train documents in a datafame format - see data_preparation
    :param test_docs: test documents in a datafame format - see data_preparation
    :param nb_iter: nb of iterations for training
    :param output_dir: directory for saving the models
    :param spacy_type: wether spacy native temporal types (DATE and TIME) or the complete types are used
    :param other_annotations: to do -> add the possibility to retrain spacy with all annotations, not just temporal ones (to avoid forgetting)
    :return:the models, the dictionary of scores
    """

    print(" ============= TRAINING MODEL ===========================")

    # tuple conversion (the tuple type is lost when dataframe -> excel -> dataframe)

    #docs['annotations'] = [[tuple(ann) for ann in annotations] for annotations in docs['annotations'].to_numpy()]


    # cross validation :

    models = []
    all_scores = []

    kf = KFold(n_splits=nb_folds)
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
                    path = 'spacy_model_' + str(c) + '_fold'
                else:
                    path = 'all_types_model_' + str(c) + '_fold'

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
                scores += [(p, r, r, pt, rt, ft)]
                print()
                print()

            # test the trained model
            test_model(val_data, nlp)

            df_scores = pd.DataFrame(scores, columns = ['span_precision', 'span_recall', 'span_f1', 'type_precision', 'type_recall', 'type_f1'])
            df_scores.to_excel(output_dir + '/' + path + '.xlsx')


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
                path = 'spacy_model_final'
            else:
                path = 'all_types_model_final'

            batches = minibatch(TRAIN_DATA, size=4)  #compounding(4.0, 20.0, 1.001)

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
            df_scores.to_excel(output_dir + '/' + path + '.xlsx')

            # save model to output directory
            if output_dir is not None:
                nlp.to_disk(output_dir + '/' + path)
                print("Saved model to", output_dir + '/' + path)

    return models, all_scores


