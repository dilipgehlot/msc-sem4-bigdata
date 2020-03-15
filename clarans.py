from pyclustering.cluster.clarans import clarans;
import pandas as pd

df=pd.read_csv('Customers.csv')

data=list(df['Age'].interpolate())


print("A peek into the dataset : ",data[:4])

clarans_instance = clarans(data, 3, 5, 4);

clarans_instance.process()

clusters = clarans_instance.get_clusters();

medoids = clarans_instance.get_medoids();

print("Index of the points that are in a cluster : ",clusters)

print("The index of medoids that algorithm found to be best : ",medoids)


#Made with ❤ By Dilip Gehlot
