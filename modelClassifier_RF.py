## importing required library for classification and model persistance
from sklearn.ensemble import RandomForestClassifier


import os
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

## classifying the authors with given classes
def get_classification(Matrix, genderList, Language, modelFolder):
    ## if not exist creating the folder to store the models
    if os.path.isdir(modelFolder) == True:
        pass
    else:
        os.makedirs(modelFolder)
    ## classification model for gender prediction
    total1 = 0
    le = preprocessing.LabelEncoder()
    genderList = le.fit_transform(genderList)

    print 'matrix size', Matrix.shape
    for i in range (0,10):
        
        X_train, X_test, y_train, y_test = train_test_split(Matrix, genderList, test_size=0.1, random_state=i)
        genderModel = RandomForestClassifier(n_estimators =100,max_depth=2, random_state=0)
        genderModel.fit(X_train, y_train)
        a = genderModel.score(X_test, y_test)
        print a
        total1 = total1 + a
    print 'gender', total1/10
    from timeit import default_timer as timer
    start = timer()
    genderModel = RandomForestClassifier(max_depth=2, random_state=0)
    genderModel.fit(X_train, y_train)
    end = timer()
    print(end - start)
    joblib.dump(genderModel, modelFolder + 'genderModel_' + Language + '.pkl')
    

## predicting the classes of the authors
def get_prediction(Matrix, Language, modelFolder):
    ## prediction of gender
    genderModel = joblib.load(modelFolder + 'genderModel_' + Language + '.pkl')
    genderPrediction = genderModel.predict(Matrix)
    #print 'Gender Prediction: ', genderPrediction
    
    return genderPrediction

