
import argparse
import os
import re
import time
import subprocess
import stat
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from distutils import util

    
def open_file(fname):
    if os.path.exists(fname):
        f = open(fname)
        return f
    else:
        outerror("No such file: %s" % fname)
        return None



def outerror(text):
    #sys.stderr.write(text + "\n")
    raise Exception(text)


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
    return id, start, end, text, type.upper(), relative, val.upper(), mod.upper()



def get_rtimex(text_fname):
    """
    This function extracts relative timexes from an annotated text
    :param text_fname: file name for the xml
    :return:the list of timexes as attributes tuples
    """
    tf = open_file(text_fname)
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
    return timexes


def compare_filtering(text_fname1, text_fname2):

    """
    This function outputs the percentage of common absolute/relative types between two annotators

    :param text_fname1: filepath for first xml file
    :param text_fname2: filepath for second xml file
    :return: a f-score ?
    """

    rtimexes1= pd.DataFrame(get_rtimex(text_fname1), columns = ['id', 'start', 'end', 'text', 'type', 'relative', 'val', 'mod']).sort_values(by = 'id')
    rtimexes2 = pd.DataFrame(get_rtimex(text_fname2), columns = ['id', 'start', 'end', 'text', 'type', 'relative', 'val', 'mod']).sort_values(by = 'id')


    if len(rtimexes1) != len(rtimexes2):
        outerror('Invalid : the number of relative time expressions is not the same')



    rel1 = [i for i in rtimexes1['relative'].to_numpy()]
    rel2 = [i  for i in rtimexes2['relative'].to_numpy()]


    t_pos = len([i for i in range(len(rel1)) if (rel1[i] and rel1[i] == rel2[i])])
    t_neg = len([i for i in range(len(rel1)) if (not rel1[i] and rel1[i] == rel2[i])])
    f_pos = len([i for i in range(len(rel1)) if (rel1[i] and rel1[i] != rel2[i])])
    f_neg = len([i for i in range(len(rel1)) if (not rel1[i] and rel1[i] != rel2[i])])

    total = len(rel1)

    f_score = f1_score(rel1, rel2, pos_label='TRUE')
    print("F1 score for the relative/absolute filtering : " + str(f_score))

    #pra = (t_pos + t_neg) / total
    #pre = (((t_pos + f_neg)/total) * ((t_pos + f_pos)/total) ) + ( ((f_pos + t_neg)/total) * ((t_neg + f_neg)/total) )
    #kappa = (pra - pre)/(1  - pre)
    kappa = cohen_kappa_score(rel1, rel2)
    print("Cohen's Kappa : " + str(kappa))

    return f_score, kappa, rel1, rel2


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
    return id, fromid, fromtext, toid, totext, relation
    


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

def is_equivalent(anchorlink1, anchorlink2, doc_annotations):


    toId1 = anchorlink1[3]
    toId2 = anchorlink2[3]

    value1 = doc_annotations[doc_annotations.id == toId1]['value'].unique()[0]
    value2 = doc_annotations[doc_annotations.id == toId2]['value'].unique()[0]

    return (value1 == value2)



def anchorlink_agreement(text_fname1, text_fname2, doc_annotations):

    """
    Evauates the agreement regarding anchorlinks between one file annotated by two annotators
    :param text_fname1: path for the first annotated file
    :param text_fname2: path for the second annotated file

    :return:
            total : total number of anchored timexes by the two annotators
            positives = nb of strictly identitcal anchorlinks
            equivalents : number fo equivalent anchor links (linking to similar anchor dates)
            negatives : number of anchorlinks truly not matching
            missing_links : number of time expressions annotated by only one annotator
    """

    links1 = pd.DataFrame(get_anchorlinks(text_fname1),
                             columns=['id', 'fromId', 'fromText', 'toId', 'toText', 'relation']).sort_values(by = 'id')
    links2 = pd.DataFrame(get_anchorlinks(text_fname2),
                             columns=['id', 'fromId', 'fromText', 'toId', 'toText', 'relation']).sort_values(by = 'id')

    print(links1)
    print()
    print(links2)

    # strict agreement
    positives = 0
    negatives = []
    equivalent = 0
    missing_links = 0 # rtimexes anchored by one annotator but not the other

    anchored_timexes = np.unique(np.concatenate([links1['fromId'].to_numpy(), links2['fromId'].to_numpy()]))


    for t_id in anchored_timexes:
        # extract the relevant timexe ?
        try :
            id, fromId, fromText, toId, toText, relation = links1[links1.fromId == t_id].to_numpy()[0]
        except Exception as e:
            print(e)
            missing_links += 1
        try:
            id2, fromId2, fromText2, toId2, toText2, relation2 = links2[links2.fromId == t_id].to_numpy()[0]
            if toId == toId2:
                positives += 1
            elif is_equivalent( (id, fromId, fromText, toId, toText, relation), (id2, fromId2, fromText2, toId2, toText2, relation2), doc_annotations):
                equivalent += 1
            else:
                negatives += [(id, id2)]
        except Exception as e:
            print(e)
            missing_links += 1

    #print('Strict agreement : ' + str(positives * 100/len(anchored_timexes)))

    print( len(anchored_timexes), positives, equivalent, len(negatives), missing_links )
    return  len(anchored_timexes), positives, equivalent, len(negatives), missing_links




'''

timexes_annotations = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')
sectimes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_sectimes_annotations.xlsx')

annotations = timexes_annotations.append(sectimes, ignore_index=True)

print(annotations)


print(anchorlink_agreement('Nicol/annotated_documents/1.xml', 'Louise/annotated_documents/1.xml', annotations[annotations.docname == '1.xml']))

'''
