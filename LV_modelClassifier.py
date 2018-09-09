## importing required library for classification and model persistance
import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

## classifying the authors with given classes
def get_classification(Matrix, languagevarietyList, Language, modelFolder):
    ## if not exist creating the folder to store the models
    if os.path.isdir(modelFolder) == True:
        pass
    else:
        os.makedirs(modelFolder)
    ## classification model for gender prediction
    total1 = 0
    le = preprocessing.LabelEncoder()
    languagevarietyList = le.fit_transform(languagevarietyList)

    print 'matrix size', Matrix.shape
    for i in range (0,10):
        
        X_train, X_test, y_train, y_test = train_test_split(Matrix, languagevarietyList, test_size=0.1, random_state=i)
        LVModel = svm.SVC()
        LVModel.fit(X_train, y_train)
        a = LVModel.score(X_test, y_test)
        print a
        total1 = total1 + a
    print 'Language Variety', total1/10
    from timeit import default_timer as timer
    start = timer()
    LVModel = svm.SVC()
    LVModel.fit(Matrix, languagevarietyList)
    end = timer()
    print(end - start)
    joblib.dump(LVModel, modelFolder + 'LVModel_' + Language + '.pkl')
    

## predicting the classes of the authors
def get_prediction(Matrix, Language, modelFolder):
    ## prediction of gender
    LVModel = joblib.load(modelFolder + 'LVModel_' + Language + '.pkl')
    LVPrediction = LVModel.predict(Matrix)
    #print 'Language Variety Predition: ', LVPrediction
    
    return LVPrediction

