# NeuralTime

This project presents the work from the published paper https://www.aclweb.org/anthology/2020.clinicalnlp-1.14/. 

It focuses on the analysis of relative and temporal time expressions (RI Timexes) in clinical text. The goal is to anchor these expressions to absolute temporal expressions, "anchor dates", to determine their time value. 

The RI_Annotations package contains utilities used to annotate the i2b2 2012 temporal dataset. 

The SVM_Anchoring package contains the scripts that were used to train SVM classifiers to anchor RI Timexes.

The BERT_Anchoring package contains the code which transforms the annotated data into suitable Bert inputs and uses it to fine-tune a BERT classifier model to perform the anchoring task. 
The pre-trained model that was used for this task is available at https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

The TemporalExtraction package contains work from an earlier project which goal was to fine-tune spacy's models to extract temporal expressions from clinical texts. 

The data that was used in this work is not made public but can be provided upon request. 

Contact the authors at : louise.dupuis97@gmail.com



