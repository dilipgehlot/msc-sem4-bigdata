
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Customers.csv')

dataset=dataset.interpolate()

X = dataset.iloc[:, [3, 4]]


# Using the elbow method to find the optimal number of clusters


ss = StandardScaler()
X = ss.fit_transform(X)
db = DBSCAN(eps=.3, min_samples=4)
db.fit(X)
y_pred = db.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=y_pred, cmap='Paired')
plt.title("DBSCAN")


#Made with ❤ By Dilip Gehlot
