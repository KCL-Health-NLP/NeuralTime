
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import ntpath
import os
from xml_utilities import escape_invalid_characters



# Goal : extract i2B2 data's annotations in the following format :
# each type of annotations (event, timex, tlink) is recorded in a separate dataframe
# Timexe annotations dataframe with : document_name, filepath, annotation_id, start, end, ann_text, type, value,
# Event annotations dataframe with : document_name, filepath, annotation_id, start, end, ann_text, modality, polarity, type
# TLink annotations dataframe with : document_name, filepath, annotation_id, fromID, fromText, toID, toText, type

# The training data files are to be found in "TimeDatasets/i2b2 Data/2012-07-15.original-annotation.release" folder
# The annotated test xml documents are in the "TimeDatasets/i2b2 Data/ground_truth" folder
#       to do : find difference between "merged" and "unmerged"
#               rename folders to make them more explicit


def extract_annotations(xml_file, document_name, test):
    """
    This function extracts timexes, events and tlinks from an xml file

    :param xml_file: xml file path to the annotated document
            document_name: the name of the document
            test : boolean, True if document is part of test set and False if part of train set
    :return: timexes : a list of timexes in tuple format : (document_name, filepath, annotation_id, start, end, type, value, ann_text,)
            events : a list of events in tuple format : (document_name, filepath, annotation_id, start, end, modality, polarity, type, ann_text,)
            tlink : a list of tlinks in tuple format : (document_name, filepath, annotation_id, fromID, fromText, toID, toText, type)
    """

    timexes = []
    tlinks = []
    events = []

    try:
        xml_string = escape_invalid_characters(xml_file)
        root = ET.fromstring(xml_string)
    except Exception as e:
        print(e)
        return None, None, None

    tags = root[1]
    print(tags)


    for ann in tags:

        # timexes annotations
        if ann.tag == 'TIMEX3':
            start = int(ann.attrib['start'])
            end = int(ann.attrib['end'])

            # we take 1 to both start and end to be consistent with other types of annotations
            if start > 0:
                start -= 1
            if end > 0:
                end -= 1

            id = ann.attrib['id']
            value = ann.attrib['val']
            type = ann.attrib['type']
            ann_text = ann.attrib['text']
            timexes += [(document_name, xml_file, id, start, end, type, value, ann_text)]

        # events annotations
        if ann.tag == 'EVENT':
            start = int(ann.attrib['start'])
            end = int(ann.attrib['end'])

            # we take 1 to both start and end to be consistent with other types of annotations
            if start > 0:
                start -= 1
            if end > 0:
                end -= 1

            id = ann.attrib['id']
            modality = ann.attrib['modality']
            polarity = ann.attrib['polarity']
            ann_text = ann.attrib['text']
            type = ann.attrib['type']
            events += [(document_name, xml_file, id, start, end, modality, polarity, type , ann_text)]

        # tlinks annotations
        if ann.tag == "TLINK":

            id = ann.attrib['id']
            fromID = ann.attrib['fromID']
            fromText = ann.attrib['fromText']
            toID = ann.attrib['toID']
            toText = ann.attrib['toText']
            type = ann.attrib['type']
            tlinks += [(document_name, xml_file, id, fromID, fromText, toID, toText, type)]



    timexes = pd.DataFrame(timexes, columns=['docname', 'text_path', 'id', 'start', 'end', 'type', 'value', 'ann_text'])
    events = pd.DataFrame(events, columns = ['docname', 'text_path', 'id', 'start', 'end', 'modality', 'polarity', 'type', 'ann_text'])
    tlinks = pd.DataFrame(tlinks, columns = ['docname', 'text_path', 'id', 'fromID', 'fromText', 'toID', 'toText', 'type'])

    timexes['test'] = [test for i in range(len(timexes))]
    events['test'] = [test for i in range(len(events))]
    tlinks['test'] = [test for i in range(len(tlinks))]

    return timexes, events, tlinks



def extract_i2b2_annotations():
    """
    This function extracts all annotations from the i2b2 database
    :return: timexes : a list of timexes in tuple format : (document_name, filepath, annotation_id, start, end, type, value, ann_text,)
            events : a list of events in tuple format : (document_name, filepath, annotation_id, start, end, modality, polarity, type, ann_text,)
            tlink : a list of tlinks in tuple format : (document_name, filepath, annotation_id, fromID, fromText, toID, toText, type)

    """

    train_data_path = '../TimeDatasets/i2b2 Data/Train-2012-07-15/'
    test_data_path =  '../TimeDatasets/i2b2 Data/Test_data/merged_xml/'

    timexes = pd.DataFrame()
    events = pd.DataFrame()
    tlinks = pd.DataFrame()

    for file_path in os.listdir(train_data_path):
        root, ext = os.path.splitext(file_path)
        if ext == '.xml':
            print(file_path)
            document_name = ntpath.basename(file_path)
            doc_timexes, doc_events, doc_tlinks = extract_annotations(train_data_path + file_path, document_name, False)

            timexes = timexes.append(doc_timexes)
            events = events.append(doc_events)
            tlinks = tlinks.append(doc_tlinks)

    for file_path in os.listdir(test_data_path):
        root, ext = os.path.splitext(file_path)
        if ext == '.xml':
            document_name = ntpath.basename(file_path)
            doc_timexes, doc_events, doc_tlinks = extract_annotations(test_data_path + file_path, document_name, True)

            timexes = timexes.append(doc_timexes, ignore_index=True)
            events = events.append(doc_events, ignore_index=True)
            tlinks = tlinks.append(doc_tlinks, ignore_index=True)

    return timexes, events, tlinks

timexes, events, tlinks = extract_i2b2_annotations()

timexes.to_excel('i2b2_timexe_annotations.xlsx')
events.to_excel('i2b2_events_annotations.xlsx')
tlinks.to_excel('i2b2_tlinks_annotations.xlsx')






