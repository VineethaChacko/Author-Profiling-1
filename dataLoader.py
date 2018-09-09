'''This scripts extracts the author ID, Language, Classes and text Data from the xml files in the data folder'''
''' Input to the load_data function is folder path where the data lies and outputs the
    genderList     --> gender class information
    authorTextList --> tweets of the authors
    languageList   --> language of the tweets written
    authorIDList   --> author's ID '''

## importing the required library to extract the data
import os
import xmlParser

def load_data(dataFolder, Action):
    truthFilePath = os.path.join(dataFolder, 'truth.txt')
    ## creating empty list for appending gender class information
    '''specific for training data '''
    genderList = [] 
    ## creating empty list for appending language of the xml file
    languageList = []
    ## creating empty list for appending author ID of the xml file
    authorIDList = []
    ## ## creating empty list for appending author's tweets from the xml file 
    authorTextList = []
    ## loop for loading training data
    if Action == 'TRAIN':
        ## reading truth file
        File = open(truthFilePath, 'r+')
        Text = File.read()
        File.close()
        truthTexts = Text.split('\n')
        ## loop for loading tweets of the author and classes from the xml files in the truth file
        for truthText in truthTexts:
            truthText = truthText.split(':::')
            if len(truthText) == 3:
                xmlFile = truthText[0] + '.xml'
                Data, Language, authorID = xmlParser.get_data(dataFolder, xmlFile)
                genderList.append(truthText[1])
                authorTextList.append(Data)
                languageList.append(Language)
                authorIDList.append(authorID)
            else:
                pass
    ## loop for loading test data
    else:
        xmlFiles = os.listdir(dataFolder)
        ## loop for loading tweets of the author from the xml files in the data folder
        for xmlFile in xmlFiles:
            Data, Language, authorID = xmlParser.get_data(dataFolder, xmlFile)
            authorTextList.append(Data)
            languageList.append(Language)
            authorIDList.append(authorID)
    return genderList, languageList, authorIDList, authorTextList
