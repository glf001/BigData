#!/usr/bin/env python
# -*-coding:utf-8 -*-
# author:GLF time:2019.11.14
# import numpy as np
# np.random.seed(4294967295)
# mu_vec1 = np.array([0,0,0])
# cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
# assert class1_sample.shape == (3,20)#检验数据的维度是否为3*20，若不为3*20，则抛出异常
#
# mu_vec2 = np.array([1,1,1])
# cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
# class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
# assert class1_sample.shape == (3,20)#检验数据的维度是否为3*20，若不为3*20，则抛出异常

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

# #############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

# #############################################################################
# Plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(np.full(n, 0.5))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'b.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()
