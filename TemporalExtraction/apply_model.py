import spacy
import pandas as pd
import numpy as np
from temporal_extractor import SpacyTemporalExtractor
from data_preparation import load_mt_samples, load_data, load_raw_data
from compute_metrics import batch_agreement
import datetime




def test_model(test_docs, nlp):

    """
    Tests a model
    :param test_docs: the documents to be tested, in a dataframe format with columns 'docname', 'annotations', and 'text'
    :param nlp: the spacy model to test
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



def test_all_folds(model_path = 'models/all_types_model/', model_name  = 'all_types_model', nb_fold = 5):
    """ Tests all five folds of a model trained with cross_validation with mt samples data

    model_path is the path to the directory were the versions of the model are saved
    model_name is the name of the model, the name of the folds should be 'model_name_i_fold' for i in range nb_fold

    :return: None
    """
    model_paths = [model_path + model_name + '_' + str(i) + '_fold' for i in range(nb_fold)]

    models = [spacy.load(model_path) for model_path in model_paths]

    metrics = []
    av_type_dict = {'DATE': None, 'FREQUENCY': None, 'DURATION': None, 'AGE_RELATED':None, 'TIME': None}

    all_annotations, documents, train_docs, test_docs = load_mt_samples()

    for models in models:
        tp_g, fp_g, fn_g, p, r, f, pt, rt, ft, type_dict = test_model(test_docs, models)
        metrics += [[tp_g, fp_g, fn_g, p, r, f, pt, rt, ft]]

        for type in type_dict.keys():
            if av_type_dict[type] is None:
                av_type_dict = type_dict

            for m in type_dict[type].keys():
                av_type_dict[type][m] = av_type_dict[type][m] + type_dict[type][m]

    for type in av_type_dict.keys():
        for m in av_type_dict[type].keys():
            av_type_dict[type][m] = av_type_dict[type][m] / 5.0

    print()
    print('============ AVERAGE RESULTS ==============================')
    print()
    print(av_type_dict)
    print()

    metrics = np.array(metrics)
    print(metrics.shape)
    metrics = metrics.sum(axis = 0) / 5.0
    print(metrics)



def apply_model_to_raw_text(path_list, model_path):

    """
    This function applies a model to raw text to extract the temporal expressions

    :param path_list: the list of path to the text file
    :param model_path: the path of the model to apply
    :return: writes a csv fils with the extracted annotations
    """

    #all_annotations, documents = load_data(annotations_path, file_path_dict)
    nlp = spacy.load(model_path)

    documents = load_raw_data(path_list)

    # this is not necessary : just to see the results
    temp_extractor = SpacyTemporalExtractor(nlp)
    results = temp_extractor.extract_expressions(documents)

    # change to sutime compatible format

    results = results.rename(columns={"doc": "doc_name"})
    results['value'] = [0 for i in range(len(results))]

    # save the output
    currentDT = str(datetime.datetime.now()).replace(':', '_').replace('.', '_').replace(' ', '_')
    print(currentDT)
    results.to_csv('results_' + currentDT + '.csv')

    #test_model(documents, nlp)

    return results

def apply_model_to_annotated_text():
    return None
