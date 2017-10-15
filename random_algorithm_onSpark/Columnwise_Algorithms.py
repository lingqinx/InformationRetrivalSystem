#-*- coding: utf-8 -*-

import random

class Columnwise_Algorithms:

	def __init__(self):
		pass

	def columnwise_multiply(self, A_matrix, B_matrix):
		if A_matrix.getColsize() != B_matrix.getRowsize():
			raise ValueError('objects are not aligned')
		all_rdd = A_matrix.getRdd().join(B_matrix.getRdd())
		result = all_rdd.mapValues(lambda x: x[0]*x[1]).map(lambda x:x[1]).reduce(lambda x, y: x+y)
		return result

	def random_multiply(self, A_matrix, B_matrix, sc, N=10):
		"""

		随机矩阵乘法
			参数
			--------
			A_matrix, B_matrix   Sparse_Matrix类
			sc                   SparkContext(用于构造广播变量)
			N                    采样次数(default=10)

		"""
		if A_matrix.getColsize() != B_matrix.getRowsize():
			raise ValueError('objects are not aligned')
		# step I
		# 计算采样概率表
		col_A_sum = A_matrix.getSquareSumByColumn()
		row_B_sum = B_matrix.getSquareSumByRow()
		p_rdd = col_A_sum.join(row_B_sum)
		p_rdd = p_rdd.mapValues(lambda x: x[0]*x[1])
		p_rdd.persist()
		p_sum = p_rdd.map(lambda x: x[1]).reduce(lambda x,y: x+y)     # 概率归一化系数
		p_dict = p_rdd.mapValues(lambda x: x/p_sum).collectAsMap()    # 概率表
		p_rdd.unpersist()

		sum_p = 0
		p_list = []   # 概率叠加表-用于快速采样算法
		for key in p_dict:
			sum_p += p_dict[key]
			one_ele = (key,sum_p)
			p_list.append(one_ele)

		def samplewithReplacement_fast(p, goal=None):
			# Sampling an Index with Replacement [faster binary-search]
			low = 0
			high = len(p)
			if goal is None:
				goal = random.random()
			if goal < p[0][1]:
				return p[0][0]
			while high > low + 1:
				middle = (low + high) / 2
				if goal >= p[middle][1]:
					low = middle
				else:
					high = middle
			return p[low + 1][0]

		# sampling
		samp_dict = {}
		for i in xrange(N):
			sample_flag = samplewithReplacement_fast(p_list)
			sample_dict[sample_flag] = sample_dict.get(sample_flag, 0) + 1


		# step II
		# 设置广播变量
		probability_dict = sc.broadcast(p_dict)
		sample_dict = sc.broadcast(samp_dict)

		def sample_by_dict(data):
			if data[0] in sample_dict.value:
				return [data] * sample_dict.value[data[0]]
			return []

		def random_multiply(data):
			return data[1][0] * data[1][1] / (N * probability_dict.value.get(data[0], 0))

		all_rdd = A_matrix.getRdd().join(B_matrix.getRdd())
		result = all_rdd.flatMap(sample_by_dict).map(random_multiply).reduce(lambda x,y : x+y)
		return result




