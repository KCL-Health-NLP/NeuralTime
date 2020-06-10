import pandas as pd


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