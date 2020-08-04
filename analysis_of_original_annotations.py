import pandas as pd
from sklearn.model_selection import train_test_split
import random

random.seed(42)

ri_original_timexes = pd.read_csv('TimeDatasets/i2b2 Data/test_reltime_gs.csv')

y_anchor = ri_original_timexes['Anchor'].to_numpy()
y_relation = ri_original_timexes['Relation_to_anchor'].to_numpy()
y_relation = [ 'After' if r == 'A' else r for r in y_relation]
# the goal is to compute statistics about these annotations that we obtained from Weiyi Sun



# we need to compute all anchor dates for each case (for example when previous absolute timex = previous timex)
anchors = ['Admission_date', 'Discharge_date', 'Previous_TIMEX', 'Previous_absolute_Timex']
anchors_dict = dict(zip(anchors, ([],[],[],[])))

for row in ri_original_timexes.to_dict('records'):
    anchor = row['Anchor']
    if anchor == 'A':
        anchor_value = row['Admission_date']
    if anchor == 'D':
        anchor_value = row['Discharge_date']
    if anchor == 'P':
        anchor_value = row['Previous_TIMEX']
    if anchor == 'PA':
        anchor_value = row['Previous_absolute_Timex']
    if anchor == 'N':
        anchor_value = ''
    for anchor in anchors:
        if anchor_value == row[anchor]:
            anchors_dict[anchor] += [True]
        else :
            anchors_dict[anchor] += [False]

ri_original_timexes['A'] = anchors_dict['Admission_date']
ri_original_timexes['D'] = anchors_dict['Discharge_date']
ri_original_timexes['P'] = anchors_dict['Previous_TIMEX']
ri_original_timexes['PA'] = anchors_dict['Previous_absolute_Timex']


# we reproduce the train/test division used in the classification process

train_data, test_data, y_anchor_train, y_anchor_test, y_relation_train, y_relation_test = train_test_split(ri_original_timexes, y_anchor, y_relation, test_size=0.15, random_state=0, stratify=y_anchor)

print(train_data)
print(test_data)


print('Train Data : Anchor relation distribution')
print(train_data.groupby(['Relation_to_anchor']).count())

print('Train set : Number of expressions anchored to the Admission Date')
print(sum(train_data['A']))
print('Train set : Number of expressions anchored to the Discharge Date')
print(sum(train_data['D']))
print('Train set : Number of expressions anchored to the Previous Timex')
print(sum(train_data['P']))
print('Train set : Number of expressions anchored to the Previous Absolute Timex')
print(sum(train_data['PA']))

print()

print('Test Data : Anchor relation distribution')
print(test_data.groupby(['Relation_to_anchor']).count())

print('Test set : Number of expressions anchored to the Admission Date')
print(sum(test_data['A']))
print('Test set : Number of expressions anchored to the Discharge Date')
print(sum(test_data['D']))
print('Test set : Number of expressions anchored to the Previous Timex')
print(sum(test_data['P']))
print('Test set : Number of expressions anchored to the Previous Absolute Timex')
print(sum(test_data['PA']))













