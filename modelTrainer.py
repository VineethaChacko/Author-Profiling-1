import os, sys, getopt
import numpy
import dataLoader
import featureLearner
import modelClassifier
import modelClassifier_RF
import modelClassifier_AB
import modelClassifier_DT

modelFolder = 'Model/'
trainFolder = '/home/vineetha/AUTHPROFILING_2017/authorprofiling.training.dataset.2017.03.10/en_3600/'
dataFolder = trainFolder
## loading the tweets from the xml file
genderList, languageList, authorIDList, authorTextList = dataLoader.load_data(dataFolder, 'TRAIN')
## temporary tweets concatenated text list per author
tempTextList = [' '.join(authorText) for authorText in authorTextList]

## 1. tdidf + SVM classifier
print '\nModel: English: tdidf + SVM'
for i in range(0,5):
    dtMatrix = featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)



##Feature Learner: Document Term Matrix

## 1. dtm + SVM classifier
print '\nModel: Arabic: dtm + SVM classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)    


## 2. dtm + Random Forest classifier
print '\nModel: Arabic: dtm + Random Forest classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_RF.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_RF.get_prediction(dtMatrix, languageList[0], modelFolder)

## 3. dtm + AdaBoost classifier
print '\nModel: Arabic: dtm + AdaBoost classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_AB.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_AB.get_prediction(dtMatrix, languageList[0], modelFolder)


## 4. dtm + Decision Tree classifier
print '\nModel: Arabic: dtm + Decision Tree classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_dtm(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_DT.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_DT.get_prediction(dtMatrix, languageList[0], modelFolder)
    
    

##Feature Learner: TFIDF

## 1. tdidf + SVM classifier
print '\nModel: Arabic: tdidf + SVM'
for i in range(0,11):
  dtMatrix = featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
  modelClassifier.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
  modelClassifier.get_prediction(dtMatrix, languageList[0], modelFolder)

## 2. tdidf + Random Forest classifier
print '\nModel: Arabic: tdidf + Random Forest classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_RF.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_RF.get_prediction(dtMatrix, languageList[0], modelFolder)

## 3. tdidf + AdaBoost classifier
print '\nModel: Arabic: tdidf + AdaBoost classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_AB.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_AB.get_prediction(dtMatrix, languageList[0], modelFolder)
    
## 4. tdidf + Decision Tree classifier
print '\nModel: Arabic: tdidf + Decision Tree classifier'
for i in range(0,11):
    dtMatrix = featureLearner.get_tdidf(tempTextList, modelFolder, languageList[0], 'TRAIN', i)
    modelClassifier_DT.get_classification(dtMatrix, genderList, languageList[0], modelFolder)
    modelClassifier_DT.get_prediction(dtMatrix, languageList[0], modelFolder)

    

