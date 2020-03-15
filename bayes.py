from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
"""
iris = load_iris()
# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target
"""

data = pd.read_csv('tennis.csv')

data=data.fillna(method='ffill')


data =data.apply(preprocessing.LabelEncoder().fit_transform)

print( data.head(15))

y = data.Play
x = data.drop('Play', axis=1)

# splitting X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# training the model on training set


gnb = GaussianNB()

g=gnb.fit(X_train, y_train)
# making predictions on the testing set
y_pred = gnb.predict(X_test)
# comparing actual response values (y_test) with predicted response values (y_pred)

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
results = confusion_matrix(y_test,y_pred)

print( 'Confusion Matrix :')
print(results)
sn.heatmap(results, annot=True)


#Made with ❤ By Dilip Gehlot
