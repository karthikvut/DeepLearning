import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

n_samples = 1000
n_outliers = 50

X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)

X[:n_outliers] = 3+0.5*np.random.normal(size=(n_outliers,1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=(n_outliers))

lr = linear_model.LinearRegression()
lr.fit(X,y)

ransac = linear_model.RANSACRegressor()
ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(X.min(), X.max())[:,np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

print("Estimated coefficients (true, linear regression, RANSAC) :")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color = 'gold', marker='.',label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth = lw, label='Linear Regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth = lw, label="Ransac Regressor")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
