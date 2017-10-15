# -*- coding:utf-8 -*-

import sys
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

class AMinerVectorizer:
	def __init__(self, max_df=0.8, min_df=20):
		self.vectorizer = TfidfVectorizer(stop_words='english',max_df=max_df,min_df=min_df,use_idf=True,norm='l2')

	# 加载文件
	def loadFile(self, filepath):            
		filefrom = open(filepath, 'r')
		try:
			all_text = filefrom.read()
			self.text = all_text.split('\n\n')    #AMiner分割符
		finally:
			filefrom.close()

	# 预处理
	def preprocessText(self):               
		processed_text = self.removeShortWord(self.text)
		processed_text = self.getLower(processed_text)
		processed_text = self.removePunctuationAndDigit(processed_text)
		processed_text = self.removeShortWord(processed_text)
		processed_text = self.removeStopWords(processed_text)
		processed_text = self.getDocsStemmed(processed_text)
		return processed_text

	# TF-IDF向量化
	def textTfidfvectorized(self, text):
		#returntype scipy.sparse.csr.csr_matrix
		return self.vectorizer.fit_transform(text)

	# 保存至文件
	def saveFile(self, matrix, filename):
		fileto = open(filename, 'w')
		matrix_temp = matrix.tocoo()
		shape = matrix_temp.shape
		fileto.write('%d\t%d\n' % (shape[0], shape[1]))
		# print '%d\t%d' % (shape[0], shape[1])
		num_data = len(matrix_temp.data)
		for i in xrange(num_data):
			fileto.write('%d\t%d\t%f\n' % (matrix_temp.row[i], matrix_temp.col[i], matrix_temp.data[i]))
			# print '%d\t%d\t%f' % (matrix_temp.row[i], matrix_temp.col[i], matrix_temp.data[i])
		fileto.close()

	# 去除标点符号
	def removePunctuationAndDigit(self, docs):
		return [re.sub('[0-9!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',doc) for doc in docs]

	# 转化为小写
	def getLower(slf, docs):
		return [doc.lower() for doc in docs]

	# 去除长度过小的单词
	def removeShortWord(self, docs):
		return [' '.join(word for word in doc.strip().split() if len(word) >= 3) for doc in docs]

	# 去除停用词
	def removeStopWords(docs):
		wordset = set(stopwords.words('english'))
		return [' '.join(word for word in doc.strip().split() if word not in wordset) for doc in docs]

	# stemming
	def getDocsStemmed(docs):
		stemmer = PorterStemmer()
		return [' '.join(stemmer.stem(word) for word in doc.strip().split()) for doc in docs]

if __name__ == '__main__':
	filefrom = sys.argv[1]
	fileto = sys.argv[2]
	vect = AMinerVectorizer()
	vect.loadFile(filefrom)
	text = vect.preprocessText()
	matrix = vect.textTfidfvectorized(text)
	vect.saveFile(matrix, fileto)
