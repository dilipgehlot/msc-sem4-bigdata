# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix 
import seaborn as sn
import graphviz
from IPython.display import display
from sklearn import tree



# load dataset
df = pd.read_csv("diabetes.csv")
print(df.head())

pima=df.fillna(method="ffill")

#split dataset in features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
results = confusion_matrix(y_test,y_pred) 
print( 'Confusion Matrix :')
print(results)
sn.heatmap(results, annot=True)



display(graphviz.Source(tree.export_graphviz(clf, None)))

#Made with ❤ By Dilip Gehlot



