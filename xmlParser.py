'''This scripts extracts the author ID, Language and text Data from the xml file'''
''' Input to the get_data function is xml file path where the data lies and outputs the
    Data     --> tweets in the xml file
    Language --> language used by the author and
    authorID --> author id of the xml file '''

## importing the required library to extract the data from xml file
import os
from xml.dom.minidom import parse
import xml.dom.minidom

def get_data(dataFolder, testFile):
    xmlFile = os.path.join(dataFolder, testFile)
    ## creating document object model from the xml file
    DOMTree = xml.dom.minidom.parse(xmlFile)
    collection = DOMTree.documentElement
    ## getting author ID of the xml file 
    authorID = testFile.replace('.xml', '')
    ## getting language of the xml file 
    Language = collection.getAttribute("lang")
    ## getting data in the xml file
    Documents = collection.getElementsByTagName("document")
    Data = []
    for Document in Documents:
        if len(Document.childNodes) > 0:
            Data.append(Document.childNodes[0].data.encode('utf-8'))
        else:
            Data.append(' ')
    return Data, Language, authorID


## for development of code
##xmlFile = '/home/vineetha/AUTHPROFILING_2017/authorprofiling.training.dataset.2017.03.10/ar_2400/ffd52f3c7e12435a724a8f30fddadd9c.xml'
#### creating document object model from the xml file
##DOMTree = xml.dom.minidom.parse(xmlFile)
##collection = DOMTree.documentElement
#### getting language of the xml file 
##Language = collection.getAttribute("lang")
#### getting data in the xml file
##Documents = collection.getElementsByTagName("document")
##Data = []
##for Document in Documents:
##    Data.append(Document.childNodes[0].data.encode('utf-8'))
