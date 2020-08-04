import pandas as pd
import matplotlib.pyplot as plt
from spacy.lang.en import English
import numpy as np

def analysis_original_data():
    original_annotations = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')
    print(original_annotations)

    print(len(original_annotations['docname'].unique()))

    date_and_time = pd.read_excel('date_and_time.xlsx')
    date_and_time = date_and_time.rename(columns={'id': 'TIMEX_id'})

    test_date_and_time = date_and_time[date_and_time['test'] == True]

    # numbers of train and test docs
    print('nb of test docs ' + str(len(test_date_and_time['docname'].unique())))
    print('nb of train docs ' + str(len(date_and_time[date_and_time['test'] == False]['docname'].unique())))
    # date_and_time['TIMEX_id'] = date_and_time['TIMEX_id'].astype('str')
    # original_annotations['TIMEX_id'] = original_annotations['TIMEX_id'].astype('str')
    s = original_annotations.merge(date_and_time, how='inner', on=['docname', 'TIMEX_id'])
    print(s)

    print('percentage of relative time expressions in the documents covered by this data: ')
    dt_nb = len(date_and_time[date_and_time['docname'].isin(original_annotations['docname'].unique())])
    print('according to these gs annotations ' + str(len(original_annotations) * 100 / dt_nb))
    print('after my first filtering ' + str(len(date_and_time[(date_and_time['docname'].isin(
        original_annotations['docname'].unique())) & (date_and_time.absolute == False)]) * 100 / dt_nb))

    timexes = pd.read_excel('i2b2_timexe_annotations.xlsx')
    print(len(timexes['docname'].unique()))
    print(len(date_and_time['docname'].unique()))


def analysis_mapped_custom_data(mapped_data):

    data_by_document = mapped_data.groupby('docname').count()['Anchor']

    data_by_anchor = mapped_data.groupby('Anchor').count()['docname']
    data_by_relation = mapped_data.groupby('Relation_to_anchor').count()['docname']

    print(data_by_document)
    print(data_by_anchor)
    print(data_by_relation)

def distance_to_anchor_date():

    """
    Plots the number of tokens between a ri timexe and its anchor date

    """


    timexes = pd.read_excel('../RI_Annotations/Results/annotated_timexes.xlsx')
    anchorlinks = pd.read_excel('../RI_Annotations/Results/anchorlinks.xlsx')

    # selecting relative timexes
    r_timexes = timexes[timexes['annotated_relative']]

    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    before = 0
    after = 0

    length = dict()

    for r in r_timexes.to_dict('records'):
        id = r['id']
        start = r['start']
        end = r['end']
        docname = r['docname']

        length[docname] = dict()
        # extract document text :

        text = open('../TimeDatasets/i2b2 Data/all_data/' + docname + '.txt').read()

        tokenized_text = tokenizer(text)
        # span = tokenized_text.char_span(start, end)
        # token = span.merge()

        # extract the anchordate

        anchors = anchorlinks[(anchorlinks['docname'] == docname) & (anchorlinks['fromID'] == id)]
        print(anchors)
        if len(anchors) == 0:
            length[docname][id] = 0
        else:
            anchor = timexes[timexes['id'] == anchors.to_dict('records')[0]['toID']].to_dict('records')
            anchor = anchor[0]
            print(anchor)
            print(start)
            if start > anchor['start']:
                before += 1
                span = tokenized_text[anchor['start']: start]
                print(span)
                length[docname][id] = len(span)

            else:
                after += 1
                span = tokenized_text[start: anchor['start']]
                print(len(span))
                length[docname][id] = len(span)

    length_flattened = []

    for doc in length.values():
        length_flattened += doc.values()

    print(length_flattened)

    thresholds = [0, 10, 50, 100, 200, 300, 400, 500, 512, 700, 1500]
    percentages = []
    for t in thresholds:
        p = [1 if l >= t else 0 for l in length_flattened].count(1) * 100 / len(length_flattened)
        percentages += [p]

    print(before)
    print(after)
    print(thresholds)
    print(percentages)

    plt.plot(thresholds, percentages)
    plt.show()

