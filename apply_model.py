import spacy
import pandas as pd
import numpy as np
from training_spacy import test_model
from temporal_extractor import SpacyTemporalExtractor
from data_preparation import load_mt_samples, load_data
from compute_metrics import batch_agreement


def test_all_folds(model_path = 'models/all_types_model/', model_name  = 'all_types_model', nb_fold = 5):
    """ Tests all five folds of a model trained with cross_validation

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



def apply_model(file_path_dict, model_path, annotations_path = None):

    all_annotations, documents = load_data(annotations_path, file_path_dict)
    nlp = spacy.load(model_path)

    # this is not necessary : just to see the results
    temp_extractor = SpacyTemporalExtractor(nlp)
    results = temp_extractor.extract_expressions(documents)
    print(results)

    test_model(documents, nlp)

    return None

test_all_folds()
