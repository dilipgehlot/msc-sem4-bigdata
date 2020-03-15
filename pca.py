from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape
breast_labels = breast.target
breast_labels.shape


import numpy as np
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape


import pandas as pd
breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels
breast_dataset.head()

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
x.shape
np.mean(x),np.std(x) # check whether the normalized data has a mean of zero and a standard deviation of one.
#Let's convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()


from sklearn.decomposition import PCA

pca_breast = PCA(n_components=2)
principalComponents_breast = pca_breast.fit_transform(x)
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
, columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

#Made with ❤ By Dilip Gehlot
