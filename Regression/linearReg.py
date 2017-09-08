import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

dataX = diabetes.data[:, np.newaxis, 2]

dataXtrain = dataX[:-50]
dataXtest = dataX[-50:]

dataYtrain = diabetes.target[:-50]
dataYtest = diabetes.target[-50:]

regr = linear_model.LinearRegression()

regr.fit(dataXtrain,dataYtrain)

dataYpred = regr.predict(dataXtest)

print("The coefficients",regr.coef_)
print("The Mean Squared Error %.2f", mean_squared_error(dataYpred, dataYtest))
print("The Variance Score .%2f", r2_score(dataYpred, dataYtest))

plt.scatter(dataXtest,dataYtest, color='black')
plt.plot(dataXtest,dataYpred, color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()

