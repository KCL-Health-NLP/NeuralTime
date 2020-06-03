import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import numpy as np


def change_annotations(xml_path, output_path, date_and_time_ann = None, tlinks = None):

    """
    This function converts an original xml i2b2 format to one suited for the RI annotation purposes

    TO DO : add tlinks to anchor links conversion (?)

    :param xml_path: the path to the original xml file
    :param output_path: the path to store the converted xml file
    :param date_and_time_ann: the absolute/relative timexes for the document (subset of date_and_time)
    :param tlinks : optionnal, the temporal tlinks to convert into anchor links
    :return: saves the updated xml file in output path
    """

    try:
        doc = minidom.parse(xml_path)
    except Exception as e:
        print(e)
        return None
        """string = escape_invalid_characters(xml_path)
        doc = minidom.parseString(string)
"""
    tags = doc.getElementsByTagName("TAGS")[0]


    # removing all timelinks and events :
    def remove_tags(tagname):
        nodes = doc.getElementsByTagName(tagname)

        for node in nodes:
            parent = node.parentNode
            parent.removeChild(node)
    remove_tags('EVENTS')
    remove_tags('TLINK')


    for ann in doc.getElementsByTagName('TIMEX3'):

        # delete non date/tume timexes
        type = ann.getAttribute('type')
        if type not in ['DATE', 'TIME']:
            parent = ann.parentNode
            parent.removeChild(ann)
        else:
            # convert timexes into absolute/relative
            id = ann.getAttribute('id')
            absolute = date_and_time_ann[date_and_time_ann.id == id]['absolute'].to_numpy()[0]
            if absolute:
                ann.tagName = 'ATIMEX3'
            else:
                ann.tagName = 'RTIMEX3'

    with open(output_path, "w") as f:
        doc.documentElement.writexml(f)



## preparing the files to be annotated


def prepare_batch(batch, name, date_and_time = pd.read_excel('../Normalization/date_and_time.xlsx'), local_path = '../TimeDatasets/i2b2 Data/Train-2012-07-15'):
    """
    this function creates a directory for annotating, with the xml files and an output folder

    :param batch: list of document names
    :param name: name for the repository to be created
    :param local_path : path to the xml documents
    :return: creates a repository, saves the modified xml files and creates an output folder for the annotated documents
    """
    os.mkdir(name)
    os.mkdir(name + '/annotated_files')

    for docname in batch:
        print(docname)

        docpath = local_path+ '/' + docname
        newpath = name + '/' + docname
        change_annotations(docpath, newpath, date_and_time[date_and_time.docname == docname])

    return None
























