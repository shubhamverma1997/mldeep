import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#reading data
data=pd.read_fwf('brain-body.txt')
x=data[['Brain']]
y=data[['Body']]

#train model

body=linear_model.LinearRegression()
body.fit(x,y)

#visualizing
plt.scatter(x,y)
plt.plot(x,body.predict(x))
plt.show()
