#-*- coding: utf-8 -*-

from pyspark import SparkConf, SparkContext
import numpy as np
import scipy.sparse as sp

class Sparse_Matrix:
	def __init__(self, M_size, content):
		self.M_size = M_size
		self.content = content

	def columnized(self):
		func_col = self.make_Mcolumn
		M_column_rdd = self.content.map(lambda x:(x[0][1],[[x[0][0],x[1]]])).reduceByKey(lambda x,y:x+y).mapValues(func_col)
		M_column_rdd.persist()   #持久化
		M_column_rdd.count()
		return M_column_rdd

	def rowized(self):
		func_row = self.make_Mrow
		M_row_rdd = self.content.map(lambda y:(y[0][0],[[y[0][1],y[1]]])).reduceByKey(lambda x,y:x+y).mapValues(func_row)
		M_row_rdd.persist()   #持久化
		M_row_rdd.count()
		return M_row_rdd

	def make_Mcolumn(self, data_list):
		row = []
		col = []
		data = []
		for element in data_list:       #构造稀疏列向量
			row.append(element[0])
			col.append(0)
			data.append(element[1])
		tempcol = sp.coo_matrix((data, (row, col)), shape=(self.M_size[0], 1)).tocsr()
		return tempcol

	def make_Mrow(self, data_list):
		row = []
		col = []
		data = []
		for element in data_list:      #构造稀疏行向量
			row.append(0)
			col.append(element[0])
			data.append(element[1])
		temprow = sp.coo_matrix((data, (row, col)), shape=(1,self.M_size[1])).tocsr()
		return temprow

class Columnwise_Algorithms:
	def __init__(self, A_size, B_size):
		self.A_size = A_size
		self.B_size = B_size

	def columnwise_multiply(self, A_matrix_rdd, B_matrix_rdd):
		all_rdd = A_column_rdd.join(B_row_rdd)
		result = all_rdd.mapValues(lambda x: x[0]*x[1]).map(lambda x:x[1]).reduce(lambda x, y: x+y)
		return result

def txt2matrix(line):
	linep = line.strip().split('\t')
	row = int(linep[0])
	col = int(linep[1])
	data = float(linep[2])
	return ((row,col),data)

if __name__ == '__main__':
	sc = SparkContext()
	matrix_path = ''         #矩阵Hadoop路径
	file_rdd = sc.textFile(matrix_path)
	size_matrix = file_rdd.filter(lambda x: len(x.strip().split('\t')) == 2).collect()[0]
	print 'The size of size_matrix is (%s, %s)' % (size_matrix[0], size_matrix[1])
	content_rdd = file_rdd.filter(lambda x: len(x.strip().split('\t')) == 3).map(txt2matrix)
	aminer_matrix = Sparse_Matrix(size_matrix, content_rdd)
	print 'The number of elements in matrix is %s' % content_rdd.count()
	
