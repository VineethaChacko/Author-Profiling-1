''' This scripts converts the list of tweets into different feature representation methods
    dtMatrix --> document term matrix
    tfidfMatrix --> term frequency - inverse document matrix '''

## importing required library for feature extraction
import os
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

modelFolder = 'Model/'
## if not exist creating the folder to store the models
if os.path.isdir(modelFolder) == True:
    pass
else:
    os.makedirs(modelFolder)
    
## function for creating document term matrix
def get_dtm(data_samples, modelFolder, Language, Action, mini_df):
    if Action == 'TRAIN':
        tf_vectorizer = CountVectorizer(min_df = mini_df)
        print 'term document matrix'
        print 'minimum document frequency', mini_df
        dtMatrix = tf_vectorizer.fit_transform(data_samples)
        joblib.dump(tf_vectorizer, modelFolder + 'tf_vectorizer_' + Language + '.pkl')
    else:
        tf_vectorizer = joblib.load(modelFolder + 'tf_vectorizer_' + Language + '.pkl')
        dtMatrix = tf_vectorizer.transform(data_samples)
    return dtMatrix

## function for creating term frequency - inverse document frequency matrix
def get_tdidf(data_samples, modelFolder, Language, Action, mini_df):
    if Action == 'TRAIN':
        print 'term document matrix'
        print 'minimum document frequency', mini_df
        tfidf_vectorizer = TfidfVectorizer(min_df = mini_df, analyzer='char', ngram_range=(3,5))
        tfidfMatrix = tfidf_vectorizer.fit_transform(data_samples)
        joblib.dump(tfidf_vectorizer, modelFolder + 'tfidf_vectorizer_' + Language + '.pkl')
    else:
        tfidf_vectorizer = joblib.load(modelFolder + 'tfidf_vectorizer_' + Language + '.pkl')
        tfidfMatrix = tfidf_vectorizer.transform(data_samples)
    return tfidfMatrix

