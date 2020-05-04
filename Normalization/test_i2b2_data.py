
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

tree = ET.parse('../TimeDatasets/i2b2 Data/2012-07-15.original-annotation.release/1.xml')


# Goal : extract i2B2 data's text and Temporal annotations in the following format :
# an annotations dataframe with : document_name, annotation : start, end, type, value, filepath

local_path = '../TimeDatasets/i2b2 Data/2012-07-15.original-annotation.release/'
train_docs = []

annotations = []  # we initialise as a list because it is much faster to append rows to a list and then build a dataframe from it



for i in range(807):
    try:
        tree = ET.parse(local_path + str(i) + '.xml')
        root = tree.getroot()
        print(root)
        train_docs += [i]

        text_path = local_path + str(i) + '.xml.txt'


        # extraction of all training
        # annotations
        tags = root[1]
        for ann in tags:
            if ann.tag == 'TIMEX3':
                start = int(ann.attrib['start'])
                end = int(ann.attrib['end'])


                # we take 1 to both start and end to be consistent with other types of annotations
                if start > 0:
                    start -= 1
                if end >0:
                    end -= 1



                value = ann.attrib['val']
                type = ann.attrib['type']
                ann_text = ann.attrib['text']
                annotations += [(i, text_path, start, end, type, value, ann_text)]

    except Exception as e:
        print(e)
        assert 1

annotations_dataframe = pd.DataFrame(annotations, columns=['docname', 'text_path', 'start', 'end', 'type', 'value', 'ann_text'])

print(annotations_dataframe)
#annotations_dataframe.to_excel('i2b2_training_annotations.xlsx')

print(root)
for child in root:
    print(child.tag, child.attrib)
    for c in child:
        if c.tag == 'TIMEX3':
            print(c.tag, c.attrib)
        if c.tag == 'SECTIME':
            print(c.tag, c.attrib)



print(train_docs)
## Applying model to i2b2 data



