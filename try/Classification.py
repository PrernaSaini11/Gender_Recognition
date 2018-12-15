import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import svm
from sklearn import tree
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

#Scaling 
def scaling(x_train):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    return x_train

#Classification
def clf(x_train,x_test,y_train,y_test):
    
    #Logisitic Regression
    clf = LogisticRegression()
    clf.fit(x_train,y_train)
    y_predicted = clf.predict(x_test)
    print("LR:",metrics.accuracy_score(y_test,y_predicted))

    # SVM
    model = svm.SVC(kernel = 'linear')
    model.fit(x_train,y_train)
    y_pred1 = model.predict(x_test)
    print("SVM:",metrics.accuracy_score(y_test,y_pred1))

    #Naive Bayes
    model = GaussianNB()
    model.fit(x_train,y_train)
    y_pred2 = model.predict(x_test)
    print("NB:",metrics.accuracy_score(y_test,y_pred2))

    #KNN
    model = neighbors.KNeighborsClassifier()
    model.fit(x_train,y_train)
    y_pred3 = model.predict(x_test)
    print("KNN:",metrics.accuracy_score(y_test,y_pred3))
    
    #Decision Tree
    model = tree.DecisionTreeClassifier()
    model.fit(x_train,y_train)
    y_pred4 = model.predict(x_test)
    print("DT:",metrics.accuracy_score(y_test,y_pred4))

path = 'D:/prerna/try_2/try/final_data.csv'
data = pd.read_csv(path)

x = data.iloc[:,0:data.shape[1] - 1].values
y = data.iloc[:,data.shape[1]-1].values
#print(x.shape)
pca = decomposition.PCA(n_components = 75)
x = pca.fit_transform(x)
print(x.shape)
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,random_state = 1,test_size = 0.3)
x_train = scaling(x_train)
clf(x_train,x_test,y_train,y_test)
