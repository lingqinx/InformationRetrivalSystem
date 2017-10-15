#-*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

class Sparse_Matrix:
	""""

	稀疏矩阵数据类型
	    成员变量
	     	rdd  RDD数据
	     	otherpart  矩阵相关操作类 

	"""

	def __init__(self, M_size, rdd, mark='element'):
		self.__rdd = rdd
		self.__otherpart = Sparse_Matrix_inner(M_size, mark)

	def ele2col(self):
		# 元素格式转化列向量
		self.__rdd = self.__otherpart.element2column(self.__rdd)
		return True

	def ele2row(self):
		# 元素格式转化行向量
		self.__rdd = self.__otherpart.element2row(self.__rdd)
		return True

	def col2ele(self):
		# 列向量转化元素格式
		self.__rdd = self.__otherpart.column2element(self.__rdd)
		return True

	def row2ele(self):
		# 行向量转化为元素格式
		self.__rdd = self.__otherpart.row2element(self.__rdd)
		return True

	def transpose(self):
		# 转置
		self.__rdd = self.__otherpart.transpose(self.__rdd)
		return True

	def getRowsize(self):
		# 获取行数
		return self.__otherpart.M_size[0]

	def getColsize(self):
		# 获取列数
		return self.__otherpart.M_size[1]

	def getMark(self):
		# 获取矩阵格式
		return self.__otherpart.mark

	def getRdd(self):
		return self.__rdd

	def persistRdd(self):
		self.__rdd.persist()
		return True

	def unpersistRdd(self):
		self.__rdd.unpersist()
		return True

	def getSquareSumByColumn(self):
		return self.__otherpart.getSquareSumByColumn(self.__rdd)

	def getSquareSumByRow(self):
		return self.__otherpart.getSquareSumByRow(self.__rdd)


class Sparse_Matrix_inner:

	"""

	矩阵相关操作类
		成员变量
			M_size  矩阵大小
			mark  矩阵表示格式 
				element  元素格式
				row      行向量
				column   列向量

	"""

	def __init__(self, M_size, mark):
		self.M_size = M_size
		self.mark = mark

	def element2column(self, rdd):
		if 'element' != self.mark:
			raise ValueError('the matrix is %s' % self.mark)

		def make_Mcolumn(data_list):
			row = []
			col = []
			data = []
			for element in data_list:       #构造稀疏列向量
				row.append(element[0])
				col.append(0)
				data.append(element[1])
			tempcol = sp.coo_matrix((data, (row, col)), shape=(self.M_size[0], 1)).tocsr()
			return tempcol

		M_column_rdd = rdd.map(lambda x:(x[0][1],[[x[0][0],x[1]]])).reduceByKey(lambda x,y:x+y).mapValues(make_Mcolumn)
		# M_column_rdd.persist()   #持久化
		# M_column_rdd.count()

		self.mark = 'column'
		return M_column_rdd


	def element2row(self, rdd):
		if 'element' != self.mark:
			raise ValueError('the matrix is %s' % self.mark)
		def make_Mrow(data_list):
			row = []
			col = []
			data = []
			for element in data_list:      #构造稀疏行向量
				row.append(0)
				col.append(element[0])
				data.append(element[1])
			temprow = sp.coo_matrix((data, (row, col)), shape=(1, self.M_size[1])).tocsr()
			return temprow

		M_row_rdd = rdd.map(lambda y:(y[0][0],[[y[0][1],y[1]]])).reduceByKey(lambda x,y:x+y).mapValues(make_Mrow)
		# M_row_rdd.persist()   #持久化
		# M_row_rdd.count()

		self.mark = 'row'
		return M_row_rdd

	def column2element(self, rdd):
		if 'column' != self.mark:
			raise ValueError('the matrix is %s' % self.mark)
		def make_Melement_col(data):
			flat_list = []
			no_column = data[0]
			temp_column = data[1].tocoo()
			for i in len(temp_column.data):
				no_row = temp_column.row[i]
				no_data = temp_column.data[i]
				flat_list.append(((no_row, no_column), no_data))
			return flat_list
		M_element_rdd = rdd.flatMap(make_Melement_col)

		self.mark = 'element'
		return M_element_rdd

	def row2element(self, rdd):
		if 'row' != self.mark:
			raise ValueError('the matrix is %s' % self.mark)
		def make_Melement_row(data):
			flat_list = []
			no_row = data[0]
			temp_row = data[1].tocoo()
			for i in len(temp_row.data):
				no_column = temp_row.col[i]
				no_data = temp_row.data[i]
				flat_list.append(((no_row, no_column), no_data))
			return flat_list
		M_element_rdd = rdd.flatMap(make_Melement_row)

		self.mark = 'element'
		return M_element_rdd

	def transpose(self, rdd):
		if 'element' == self.mark:
			temp_rdd = rdd.map(lambda y:((y[0][1], y[0][0]), y[1]))
		elif 'column' == self.mark:
			temp_rdd = rdd.mapValues(lambda x: x.transpose())
			self.mark = 'row'
		elif 'row' == self.mark:
			temp_rdd = rdd.mapValues(lambda x: x.transpose())
			self.mark = 'column'

		self.M_size = (self.M_size[1], self.M_size[0])
		return temp_rdd

	def getSquareSumByColumn(self, rdd):
		if 'column' == self.mark:
			result_dict = rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))
		elif 'row' == self.mark:
			ele_rdd = self.row2element(rdd)
			col_rdd = self.element2column(ele_rdd)
			result_dict = col_rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))
		elif 'element' == self.mark:
			col_rdd = self.element2column(rdd)
			result_dict = col_rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))
		return result_dict

	def getSquareSumByRow(self, rdd):
		if 'row' == self.mark:
			result_dict = rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))
		elif 'column' == self.mark:
			ele_rdd = self.column2element(rdd)
			row_rdd = self.element2row(ele_rdd)
			result_dict = row_rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))
		elif 'element' == self.mark:
			row_rdd = self.element2row(rdd)
			result_dict = row_rdd.mapValues(lambda x: np.sqrt(x.multiply(x).sum()))

		return result_dict






