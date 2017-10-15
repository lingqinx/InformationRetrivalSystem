# coding:utf-8
__author__ = 'songmingye'

import numpy as np

def readFromList(path,filenames):
    docs = []
    for name in filenames:
        filefrom = open('%s/%s' % (path,name), 'r')
        docs.append(filefrom.read())
        filefrom.close()
    docs = [doc for doc in docs if doc.strip() != '']
    return docs

def readFromName(path,filename,splitsign):
    filefrom = open('%s/%s' % (path,filename), 'r')
    allfile = filefrom.read()
    docs = allfile.strip().split(splitsign)
    docs = [doc for doc in docs if doc.strip() != ''] #去除空文档
    return docs

def writeMatrixToFile(path,filename,matrix):
    np.savetxt('%s/%s' % (path,filename),matrix,fmt='%.6f',delimiter=',')

if __name__ == '__main__':
    text = readFromName('/Users/songmingye/Documents/SearchEngine','AMiner-Author.txt','\n\n')
    print len(text)