# coding:utf-8
__author__ = 'songmingye'

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.porter import PorterStemmer

#去除标点符号
def removePunctuationAndDigit(docs):
    return [re.sub('[0-9!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',doc) for doc in docs]

#去除长度过小的单词
def removeShortWord(docs):
    return [' '.join(word for word in doc.strip().split() if len(word) >= 3) for doc in docs]

#转化为小写
def getLower(docs):
    return [doc.lower() for doc in docs]

#去除停用词
def removeStopWords(docs):
    wordset = set(stopwords.words('english'))
    return [' '.join(word for word in doc.strip().split() if word not in wordset) for doc in docs]

#stemming
def getDocsStemmed(docs):
    stemmer = PorterStemmer()
    return [' '.join(stemmer.stem(word) for word in doc.strip().split()) for doc in docs]
    #stemmer = EnglishStemmer()
    #return [' '.join(stemmer.stem(word) for word in doc.strip().split()) for doc in docs]
    #whl = WordNetLemmatizer()
    #return [' '.join(whl.lemmatize(word) for word in doc.strip().split()) for doc in docs]
