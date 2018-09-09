import os, sys, getopt
import numpy
import LV_dataLoader
import LV_featureLearner
import LV_modelClassifier
import LV_modelClassifier_RF
import LV_modelClassifier_AB
import LV_modelClassifier_DT

modelFolder = 'Model/'
trainFolder = '/home/vineetha/AUTHPROFILING_2017/authorprofiling.training.dataset.2017.03.10/ar_2400/'
dataFolder = trainFolder
## loading the tweets from the xml file
languagevarietyList, languageList, authorIDList, authorTextList = LV_dataLoader.load_data(dataFolder, 'TRAIN')
## temporary tweets concatenated text list per author
tempTextList = [' '.join(authorText) for authorText in authorTextList]

## 1. tfidf + SVM classifier
print '\nArabic: Model: dtm + SVM classifier'
#for i in range(11,16):
dtMatrix = LV_featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', 10)
LV_modelClassifier.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
LV_modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)

modelFolder = 'Model/'
trainFolder = '/home/vineetha/AUTHPROFILING_2017/authorprofiling.training.dataset.2017.03.10/en_3600/'
dataFolder = trainFolder
## loading the tweets from the xml file
languagevarietyList, languageList, authorIDList, authorTextList = LV_dataLoader.load_data(dataFolder, 'TRAIN')
## temporary tweets concatenated text list per author
tempTextList = [' '.join(authorText) for authorText in authorTextList]


##Feature Learner: TFIDF
## 1. tdidf + SVM classifier
print'\nPortugese: Model: tfidf + SVM classifier'
#for i in range(0,11):
dtMatrix = LV_featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', 2)
LV_modelClassifier.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
LV_modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)

## 2. tdidf + Random Forest classifier
print '\nPortugese: Model: tdidf + Random Forest classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_RF.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_RF.get_prediction(dtMatrix, languageList[0], modelFolder)

## 3. tdidf + AdaBoost classifier
print '\nPortugese: Model: tdidf + AdaBoost classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_AB.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_AB.get_prediction(dtMatrix, languageList[0], modelFolder)
    
## 4. tdidf + Decision Tree Classifier    
print '\nPortugese: Model: tdidf + Decision Tree classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_DT.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_DT.get_prediction(dtMatrix, languageList[0], modelFolder)

    
##Feature Learner: DTM
## 1. dtm + SVM classifier
print '\nPortugese: Model: dtm + SVM classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)


## 2. dtm + Random Forest classifier
print '\nPortugese: Model: dtm + Random Forest classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_RF.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_RF.get_prediction(dtMatrix, languageList[0], modelFolder)

## 3. dtm + AdaBoost classifier
print '\nPortugese: Model: dtm + AdaBoost classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_AB.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_AB.get_prediction(dtMatrix, languageList[0], modelFolder)

## 4. dtm + Decision Tree classifier
print '\nPortugese: Model: dtm + Decision Tree classifier'
for i in range(0,11):
    dtMatrix = LV_featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    LV_modelClassifier_DT.get_classification(dtMatrix, languagevarietyList, languageList[0], modelFolder)
    LV_modelClassifier_DT.get_prediction(dtMatrix, languageList[0], modelFolder)
    
