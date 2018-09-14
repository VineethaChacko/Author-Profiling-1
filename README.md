# Author-Profiling-1
Gender and Language Variety Identification is performed on Twitter Data (training corpora of PAN@CLEF 2017 Author Profiling shared task), using Machine Learning.

# Dataset - Training Data of PAN@CLEF 2017 Author Profiling shared task - Twitter Data in English, Spanish, Arabic and Portuguese.
The dataset contains Tweets in four languages, namely, English, Spanish, Arabic and Portuguese. For each language, there is a balance in the statistics of authors, as far as gender as well as langugae variety is concerned. The data is given as XML files, each containing 100 sentences.
3600 authors in English, 4200 authors in Spanish, 2400 authors in Arabic and 1200 authors in Portuguese.

# Model
The XML data is converted to text format. 
It is then represented as Term Document Matrix and as Term Frequency - Inverse Document Frequency Matrix. 
The vocabulary (or terms) of these matrices is governed by the paramaters minimum document frequency, word n-gram, character n-gram and sub-linear term frequency. 
Further, using scikit learn, Machine Learning algorithms like SVM, Decision Tree, Random Forest and AdaBoost have been used for the classification.
10 fold cross validation accuracy is recorded.
