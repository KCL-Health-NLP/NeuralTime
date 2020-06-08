
import random
import pandas as pd
import svm_anchoring
import map_custom_annotations
import data_analysis
import os
random.seed(42)



# training anchor classififation models with the original data

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')

date_and_time = pd.read_excel('../TimeDatasets/i2b2 Data/date_and_time.xlsx')  # for now, original filtering
all_timexes = pd.read_excel('../TimeDatasets/i2b2 Data/i2b2_timexe_annotations.xlsx')

models = svm_anchoring.svm_anchoring(ri_original_timexes, date_and_time, all_timexes)


# mapping our custom data to the original format

# test, only on two files
filepaths = ['../RI_Annotations/Batches/Louise_Test/annotated_files/53.xml', '../RI_Annotations/Batches/Louise_Test/annotated_files/61.xml']

filepath_test= [ '../RI_Annotations/Batches/Louise_Test/annotated_files/' + docname  for docname in os.listdir('../RI_Annotations/Batches/Louise_Test/annotated_files') ]

print(filepath_test)
anchorlinks, timexes = map_custom_annotations.annotated_files_to_dataframe(filepath_test)


mapped_data = map_custom_annotations.custom_to_standard(anchorlinks, timexes, all_timexes)

data_analysis.analysis_mapped_custom_data(mapped_data)