
import random
import pandas as pd
import svm_anchoring
random.seed(42)



# training anchor classififation models

ri_original_timexes = pd.read_csv('../TimeDatasets/i2b2 Data/test_reltime_gs.csv')

models = svm_anchoring.svm_anchoring(ri_original_timexes)


