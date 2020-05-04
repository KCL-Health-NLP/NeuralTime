
import re


def escape_invalid_characters(xml_path):

    """
    This function escapes the invalid characters found in the i2b2 xml files
    (for now, only &)
    it returns an xml string
    :param xml_path: path to the xml file
    :return: the xml formatted string with escaped "&"
    """

    string = open(xml_path, 'r').read()

    regex = re.compile(r"&(?!amp;|lt;|gt|apos;)")
    regex2 = re.compile('&apos;') # invalid apostrophe escape
    valid_xml = regex.sub("&amp;", string)
    valid_xml = regex2.sub('_apos;', valid_xml)
    #valid_xml = regex2.sub("'", valid_xml)
    return valid_xml


escape_invalid_characters('../TimeDatasets/i2b2 Data/Train-2012-07-15/143.xml')