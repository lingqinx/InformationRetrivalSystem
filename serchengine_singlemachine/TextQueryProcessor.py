# coding:utf-8
__author__ = 'songmingye'

import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse.linalg as lins

import DocPreprocessor as dp

class TextQueryProcessor:
    def __init__(self,docs,tol=0,query_type='full',max_df=0.5,min_df=1,k=6):
        self.tol = tol
        self.docs = docs #保存原文本
        self.query_type = query_type
        self.tfidftrans = TfidfVectorizer(stop_words='english',max_df=max_df,min_df=min_df,use_idf=True,norm='l2')
        time0 = time.time()
        docs_copy = self.preprocesssing(docs)
        print 'Preprocess cost %.3f second' % (time.time() - time0)

        time0 = time.time()
        self.matrix = self.fit_transform(docs_copy)  #TFIDF向量化
        print 'TfIdf cost %.3f second' % (time.time() - time0)
        print self.matrix.shape
        #print type(self.matrix)
        if 'full' == query_type: #全向量空间模型
            pass
        if 'lsi' == query_type: #LSI模型
            self.k = k
            time0 = time.time()
            U,s,Vt = lins.svds(self.matrix.transpose(),k)
            print 'SVD cost %.3f second' % (time.time() - time0)
            self.U = U
            self.H = self.matrix.dot(U)
            #self.H = U.transpose().dot(self.matrix.transpose().toarray()).transpose()
            num = self.H.shape[1]
            #保证H的行向量为单位向量
            column = np.sqrt(np.add.reduce(self.H**2,axis=1).reshape(-1,1))
            column[column == 0] = 1  #防止可能的范数为0
            self.H = self.H/column.repeat(num,axis=1)


    def preprocesssing(self,docs):
        #去除标点符号
        docs_temp = dp.removePunctuationAndDigit(docs)
        #去除停用词
        docs_temp = dp.removeStopWords(docs_temp)
        #Stemming WordNet
        try:
            docs_temp = dp.getDocsStemmed(docs_temp)
        except:
            pass
        return docs_temp

    #def setdocs(self,docs):
        #self.docs = docs

    def settol(self,tol):
        self.tol = tol

    #returntype scipy.sparse.csr.csr_matrix
    def fit_transform(self,text):
        return self.tfidftrans.fit_transform(text)

    def transform(self,query):
        return self.tfidftrans.transform(query)

    def get_feature_names(self):
        return self.tfidftrans.get_feature_names()

    def querying(self,query):
        vect = self.transform(self.preprocesssing(query))
        time0 = time.time()
        #print vect.shape
        if 'full' == self.query_type: #全向量空间检索
            result = self.matrix.dot(np.transpose(vect))
            suitnum = result[result > self.tol].shape[1]
            resultarg = np.argsort(-result.toarray(),axis=0)[0:suitnum].T
        if 'lsi' == self.query_type:
            vect = vect.dot(self.U)
            result = self.H.dot(np.transpose(vect))
            suitnum = result[result > self.tol].shape[0]
            resultarg = np.argsort(-result,axis=0)[0:suitnum].T
        print 'Query cost %.3f second' % (time.time() - time0)
        return resultarg

