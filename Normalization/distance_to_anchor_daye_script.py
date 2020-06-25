import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spacy.lang.en import English

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
    #span = tokenized_text.char_span(start, end)
    #token = span.merge()

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
            span = tokenized_text[anchor['start']: start ]
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
    p = [1 if l>=t else 0 for l in length_flattened].count(1) * 100 / len(length_flattened)
    percentages += [p]

print(before)
print(after)
print(thresholds)
print(percentages)

plt.plot(thresholds, percentages)
plt.show()




