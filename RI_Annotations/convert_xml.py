import pandas as pd
from Normalization.xml_utilities import escape_invalid_characters
import xml.etree.ElementTree as ET
from xml.dom import minidom


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

    doc = minidom.parse(xml_path)
    tags = doc.getElementsByTagName("TAGS")[0]

    print(tags)
    print(doc.getElementsByTagName('TIMEX3'))


    # removing all timelinks and events :
    def remove_tags(tagname):
        nodes = doc.getElementsByTagName(tagname)

        for node in nodes:
            parent = node.parentNode
            parent.removeChild(node)
    remove_tags('EVENTS')
    remove_tags('TLINK')


    for ann in doc.getElementsByTagName('TIMEX3'):

        print(ann)

        # delete non date/tume timexes
        type = ann.getAttribute('type')
        if type not in ['DATE', 'TIME']:
            parent = ann.parentNode
            parent.removeChild(ann)
        else:
            # convert timexes into absolute/relative
            id = ann.getAttribute('id')
            print(id)
            print(date_and_time_ann[date_and_time_ann.id == id])
            absolute = date_and_time_ann[date_and_time_ann.id == id]['absolute'].to_numpy()[0]
            if absolute:
                ann.tagName = 'ATIMEX3'
            else:
                ann.tagName = 'RTIMEX3'

    with open(output_path, "w") as f:
        doc.documentElement.writexml(f)



date_and_time = pd.read_excel('../Normalization/date_and_time.xlsx')
docname = '311.xml'

change_annotations(docname, '311_new.xml', date_and_time[date_and_time.docname == docname])




