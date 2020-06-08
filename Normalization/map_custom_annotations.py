

"""
The goal of this file is to map the custom anchor annotations to the format of the original paper
"""

import pandas as pd
import numpy as np
import re
import os
from RI_Annotations import agreementEvaluation
from Normalization import svm_anchoring



def get_rtimex_attr(timex_line):
    """
    This function extracts attributes from a relative timex
    Args:
      line - str: MAE RTIMEX3 tag line,
                  e.g. <RTIMEX3 id="T8" start="1259" end="1268" text="that time" type="DATE" relative="TRUE" val="1992-10" mod="NA" />
    """

    re_exp = 'id=\"([^"]*)\"\s+start=\"([^"]*)\"\s+end=\"([^"]*)\"\s+text=\"([^"]*)\"\s+type=\"([^"]*)\"\s+relative=\"([^"]*)\"\s+val=\"([^"]*)\"\s+mod=\"([^"]*)\"\s+\/>'
    m = re.search(re_exp, timex_line)
    if m:
        id, start, end, text, type, relative, val, mod = m.groups()
    else:
        raise Exception("Malformed Timex3 tag: %s" % (timex_line))

    return [id, start, end, text, type.upper(), True, val.upper(), mod.upper(), (relative == 'TRUE')]



def get_atimex_attr(timex_line):
    """
    This function extracts attributes from an absolute timex
    Args:
      line - str: MAE ATIMEX3 tag line,
                  e.g. <ATIMEX3 id="T8" start="1259" end="1268" text="12/05" type="DATE"  val="1992-10" absolute = "TRUE"  mod="NA" />
    """
    re_exp = 'id=\"([^"]*)\"\s+start=\"([^"]*)\"\s+end=\"([^"]*)\"\s+text=\"([^"]*)\"\s+type=\"([^"]*)\"\s+val=\"([^"]*)\"\s+absolute=\"([^"]*)\"\s+mod=\"([^"]*)\"\s+\/>'
    m = re.search(re_exp, timex_line)
    if m:
        id, start, end, text, type, val, absolute, mod = m.groups()
    else:
        raise Exception("Malformed Timex3 tag: %s" % (timex_line))

    return [id, start, end, text, type.upper(), False, val.upper(), mod.upper(), (absolute == 'FALSE')]



def get_sectimex_attr(timex_line):
    """
    This function extracts attributes from a sectime timex
    Args:
      line - str: MAE SECTIME tag line,
                  e.g. <SECTIME id="S0" start="18" end="28" text="2010-06-28" type="ADMISSION" dvalue="2010-06-28" />
    """
    re_exp = 'id=\"([^"]*)\"\s+start=\"([^"]*)\"\s+end=\"([^"]*)\"\s+text=\"([^"]*)\"\s+type=\"([^"]*)\"\s+dvalue=\"([^"]*)\"\s+\/>'
    m = re.search(re_exp, timex_line)
    if m:
        id, start, end, text, type, dvalue = m.groups()
    else:
        raise Exception("Malformed Timex3 tag: %s" % (timex_line))
    return [id, start, end, text, type.upper(), False, dvalue, None, False]



def get_timexes(text_fname):

    """
    This function extracts r/a timexes from an annotated text
    :param text_fname: file name for the xml
    :return:the list of timexes as attributes tuples
    """

    tf = agreementEvaluation.open_file(text_fname)
    lines = tf.readlines()
    timexes = []
    # fix for ruling out duplicate ids
    unique_ids = []
    for line in lines:
        if re.search('<RTIMEX3', line):
            timexTuple = get_rtimex_attr(line)
            if timexTuple[0] not in unique_ids:
                timexes.append(timexTuple)
                unique_ids.append(timexTuple[0])
        if re.search('<ATIMEX3', line):
            timexTuple = get_atimex_attr(line)
            if timexTuple[0] not in unique_ids:
                timexes.append(timexTuple)
                unique_ids.append(timexTuple[0])
        if re.search('<SECTIME', line):
            timexTuple = get_sectimex_attr(line)
            if timexTuple[0] not in unique_ids:
                timexes.append(timexTuple)
                unique_ids.append(timexTuple[0])
    return timexes



def attr_by_line(linkline):
    """
    This function takes an MAE file ANCHORLINK line and extracts its attributes
    Args:
      line - str: MAE ANCHORLINK tag line,
                  e.g. <ANCHORLINK id="AN0" fromID="T4" fromText="that time" toID="T3" toText="1991" relation="EQUAL" />
    """
    re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+relation=\"([^"]*)\"\s+\/>'
    m = re.search(re_exp, linkline)
    if m:
        id, fromid, fromtext, toid, totext, relation = m.groups()
    else:
        raise Exception("Malformed EVENT tag: %s" % (linkline))
    return [id, fromid, fromtext, toid, totext, relation]



def get_anchorlinks(text_fname):
    '''
    This function extracts the anchor links from an annotated xml file
    Args:
        text_fname: file name of the MAE xml file

    Output:
        a tlinks tuple of all the tlinks in the file
    '''
    tf=open(text_fname) # why the -5 ?
    lines = tf.readlines()
    anchorlinks=[]
    for line in lines:
        if re.search('<ANCHORLINK',line):
            anchorlink_tuple=attr_by_line(line)
            anchorlinks.append(anchorlink_tuple)
    return anchorlinks



def annotated_files_to_dataframe(filepaths):

    """
    Transforms a group of annotated xml files into two dataframe tables with annotations information
    :return: anchorlinks : a table with the anchor links
            timexes : a table with all the relative and absolute timexes
    """

    anchorlinks = []
    timexes = []
    for path in filepaths:
        docname = os.path.basename(path)

        doc_anchorlinks = get_anchorlinks(path)

        if len(doc_anchorlinks)>0:
            doc_anchorlinks = np.append(np.array([[docname] for i in range(len(doc_anchorlinks))]), np.array(doc_anchorlinks), axis=1).tolist()
        doc_timexes = get_timexes(path)
        doc_timexes = np.append(np.array([[docname] for i in range(len(doc_timexes))]), np.array(doc_timexes), axis=1 ).tolist()


        anchorlinks += doc_anchorlinks
        timexes += doc_timexes

    anchorlinks = pd.DataFrame(anchorlinks, columns = ['docname', 'id', 'fromID', 'fromText', 'toID', 'toText', 'relation'])
    timexes = pd.DataFrame(timexes, columns = ['docname','id', 'start', 'end',' text', 'type','filtered_relative', 'val', 'mod', 'annotated_relative' ])

    print(anchorlinks)
    print(timexes)

    return anchorlinks, timexes



def custom_to_standard(anchorlinks, timexes, all_timexes):

     print('Custom to standard')

     """
     Takes the anchorlink and timexe dataframes and produces a dataframe with the same format as the standard annotations we received

     :return:
     standard_df = a dataframe with the following format : docname, TIMEX_id, TIMEX_value, TIMEX_text, Admission_date, Discharge_date, Previous_timex, Previous_absolute_timex, Anchor, Relation_to_anchor
     """

     timexes['absolute'] = [not relative for relative in timexes['annotated_relative']]
     print(timexes)


     RI = timexes[timexes['annotated_relative']]

     print('RI')
     print(RI)

     def admission_and_discharge_ids(docname):

         doc_timexes = timexes[timexes.docname == docname]



         # extract admission and discharge ids

         admission = doc_timexes[doc_timexes.type == 'ADMISSION'].to_dict('records')[0]
         admission_id = doc_timexes[doc_timexes.start == admission['start']].to_dict('records')[0]['id']
         try :
             discharge = doc_timexes[doc_timexes.type == 'DISCHARGE'].to_dict('records')[0]
             discharge_id = doc_timexes[doc_timexes.start == discharge['start']].to_dict('records')[0]['id']
         except Exception as e:
             print(e)
             discharge_id = 'S1'
         return admission_id, discharge_id








     def extract_ids(docname, id):

         """
         Gets ids for  Admission_date, Discharge_date, Previous_timex, Previous_absolute_timex
         :return:
         """

         previous_id, previous_absolute_id = svm_anchoring.get_previous_timexes(id, docname, timexes, all_timexes)

         admission_id, discharge_id = admission_and_discharge_ids(docname)

         return admission_id, discharge_id, previous_id, previous_absolute_id


     def get_anchor(docname, id):

         admission_id, discharge_id, previous_id, previous_absolute_id = extract_ids(docname, id)
         link = anchorlinks[(anchorlinks.docname == docname) & (anchorlinks.fromID == id)].to_dict('records')

         if len(link) > 0:
             toID = link[0]['toID']
             relation = link[0]['relation'][0]

             if toID == admission_id or toID == 'S0':
                 anchor = 'A'
             elif toID == discharge_id or toID == 'S1':
                 anchor = 'D'
             elif toID == previous_absolute_id:
                 anchor = 'PA'
             elif toID == previous_id:
                 anchor = 'P'
             else:
                 anchor = 'O'


         else :
             anchor = 'N'
             relation = 'NA'

         return [docname, id, admission_id, discharge_id, previous_id, previous_absolute_id, anchor, relation]

     result = [get_anchor(docname, id) for docname, id in zip(RI['docname'], RI['id'])]
     result = pd.DataFrame(result, columns=['docname', 'TIMEX_id',  'Admission_id', 'Discharge_id', 'Previous_id','Previous_absolute_id', 'Anchor', 'Relation_to_anchor'])

     print(result)
     return result















