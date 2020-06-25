
import  pandas as pd


def convert_to_tuple_modelling():
    """
    This function aims at converting our RITimexes and anchorlinks to a list of (RTimexe, Timexe) pairs where the potential anchoring and
    anchor relation is specified.
    """

    timexes = pd.read_excel('../RI_Annotations/Results/annotated_timexes.xlsx')
    anchorlinks = pd.read_excel('../RI_Annotations/Results/anchorlinks.xlsx')

    # selecting relative timexes
    r_timexes = timexes[timexes['annotated_relative']]








convert_to_tuple_modelling()


