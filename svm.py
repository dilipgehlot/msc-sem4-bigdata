
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics

# Split dataset into training set and test set

cancer = datasets.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets

clf.fit(X_train, y_train)
#Predict the response for test dataset

y_pred = clf.predict(X_test)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='spring');
plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
results = confusion_matrix(y_test,y_pred)
print( 'Confusion Matrix :')
print(results)
sn.heatmap(results, annot=True)

#Made with ❤ By Dilip Gehlot
