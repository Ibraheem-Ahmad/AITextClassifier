Hey Brian
Here's a breakdown of how the code works

1. ExplorAnalysis+FeatureCreation.py is the first file for some inital exploraty analysis
I also use it for created the engineered features

2. NonTextFeatureClassifiers.py is the file i used for training models on just the engineerred features
not the actual text itself. 

3. TDFVectorClassifiers.py 
THis is classifiers using TDF Vectorization for text, it does not have the engineered features

4. CustomFeatuers+TDF.py this file is has the custom features plus TDF vecotirzastions
but we might need to do feature selectino with this cause it has a lot of features to make it better
not sure

5.HyperParemeterTuning.py pretty self explanatory the two model hyper paremeter tuning.
Probably could make this easier