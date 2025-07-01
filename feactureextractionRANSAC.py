import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt


np.random.seed(0)
x = np.random.uniform(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

outliers_index = np.random.choice(100, 20, replace=False)
y[outliers_index] += 10 * np.random.normal(0, 1, 20)


data = np.vstack((x, y)).T


ransac = RANSACRegressor()


ransac.fit(data[:, 0].reshape(-1, 1), data[:, 1])


inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)


line_slope = ransac.estimator_.coef_[0]
line_intercept = ransac.estimator_.intercept_


plt.scatter(data[inlier_mask][:, 0], data[inlier_mask][:, 1], c='b', label='Inliers')
plt.scatter(data[outlier_mask][:, 0], data[outlier_mask][:, 1], c='r', label='Outliers')


plt.plot(x, line_slope * x + line_intercept, color='g', label='RANSAC line')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

