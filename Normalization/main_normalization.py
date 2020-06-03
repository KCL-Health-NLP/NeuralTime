
import random
import pandas as pd
import svm_anchoring
import map_custom_annotations
random.seed(42)



# training anchor classififation models with the original data

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')

models = svm_anchoring.svm_anchoring(ri_original_timexes)


# mapping our custom data to the original format

# test, only on two files
filepaths = ['../RI_Annotations/Batches/Louise_Test/annotated_files/53.xml', '../RI_Annotations/Batches/Louise_Test/annotated_files/61.xml']

anchorlinks, timexes = map_custom_annotations.annotated_files_to_dataframe(filepaths)

print(timexes['annotated_relative'])
map_custom_annotations.custom_to_standard(anchorlinks, timexes)

