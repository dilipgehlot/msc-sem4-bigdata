from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt


df=pd.read_csv("Customers.csv.")

df=df.interpolate()

dataset=df.iloc[:, [3, 4]].values

print(dataset)



km = KMeans(n_clusters=3)
km.fit(dataset)

centroids=km.cluster_centers_
print("center:--------------------\n",centroids)

y_pred = km.predict(dataset)
plt.scatter(dataset[:,0], dataset[:,1],c=y_pred, cmap='Paired')
plt.title("K-means")

#Made with ❤ By Dilip Gehlot
