import pandas as pd
import re

# gather data

""" This file contains the necessary code to perform the first filtering to differentiate absolute from relative timexes."""
# extract relative timexes


#time_tlinks = extract_ri_timexes()

def is_absolute_timexe(string):

    """
    This function uses patterns to test wether or not a temporal annotation is of a known absolute format
    :param string: the annotation
    :return: boolean
    """
    patterns = [
        '(\d+)', # just digits
        '(\d+/\d+/\d+)',
        '(\d+/\d+)',
        '(\d+-\d+-\d+)',
        '(\d+-\d+)',
        '^\d{1,2}\/\d{1,2}\/\d{4}$',  # matches dates of the form XX/XX/YYYY where XX can be 1 or 2 digits long and YYYY is always 4 digits long.
        "^((([0]?[1-9]|1[0-2])(:|\.)[0-5][0-9]((:|\.)[0-5][0-9])?( )?(AM|am|aM|Am|PM|pm|pM|Pm))|(([0]?[0-9]|1[0-9]|2[0-3])(:|\.)[0-5][0-9]((:|\.)[0-5][0-9])?))$",  # Matches times seperated by either : or . will match a 24 hour time, or a 12 hour time with AM or PM specified. Allows 0-59 minutes, and 0-59 seconds. Seconds are not required.
        "^ ((0[1 - 9]) | (1[0 - 2]))\ / (\d{2})$"
    ]

    for pattern in patterns:
        if re.match(pattern, string) is not None:
            return True

    return False




def filter_absolute_timexes():

    """
    this function filters absolute timexes using the patterns above
    it saves the resulting tables as :
    - absolute_timexes
    - filtered_timexes for the relative timexes
    - date_and_time for all the date and time timexes, with a boolean attribute : "absolute"
    :return: relative_timexes
    """


    timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')

    timexes = timexes[timexes['type'].isin(['DATE', 'TIME'])]

    print('DATE AND TIME')
    print(timexes)
    absolute_timexes = timexes[ [is_absolute_timexe(string) for string in timexes['ann_text']] ]

    print('ABSOLUTE TIMEXES')
    print(absolute_timexes)

    absolute_timexes.to_excel('absolute_timexes.xlsx')
    relative_timexes = timexes[[(not is_absolute_timexe(string)) for string in timexes['ann_text']]]

    relative_timexes.to_excel('filtered_timexes.xlsx')

    # add the absolute characteristic as a boolean attribute of the timexe dataframe

    timexes['absolute'] = [is_absolute_timexe(string) for string in timexes['ann_text']]

    timexes.to_excel('date_and_time.xlsx')

    print('RELATIVE TIMEXES')
    print(relative_timexes)

    # Print the results

    print(len(timexes[(timexes.absolute == False) & (timexes.test == False)]['docname'].unique()))

    train_relatives = timexes[(timexes.absolute == False) & (timexes.test == False)]
    test_relatives = timexes[(timexes.absolute == False) & (timexes.test == True)]

    print('Train set : ' + str(len(train_relatives)) + " relative time expressions")
    print('Test set : ' + str(len(test_relatives)) + " relative time expressions")

    return relative_timexes


#filter_absolute_timexes()

