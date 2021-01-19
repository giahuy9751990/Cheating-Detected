import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def gmm_build_model(X_Train,n_components = 2):
  from sklearn.mixture import GaussianMixture
  gmm = GaussianMixture(n_components=n_components)
  gmm.fit(X_train)
  return gmm

def gmm_chon_k_components(X_train , begin = 2, end = 5):
  from sklearn.mixture import GaussianMixture
  from sklearn import metrics
  list_sil = []
  K = range(begin,end)
  for k in K:
    gmm = GaussianMixture(n_components=k)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    sil = metrics.silhouette_score(X_train,labels,metric='euclidean')
    list_sil.append(sil)
  plt.plot(K,list_sil,'bx-')
  plt.xlabel('k')
  plt.ylabel('sil_score')
  plt.title("The silhouette_score & k")
  plt.show()

def hierarchy(data,number_cluster,affinity="euclidean",linkage='ward'):
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=number_cluster,affinity="euclidean",linkage='ward')
    cluster.fit(data)
    return cluster

def ve_dengrogram(data):
    import scipy.cluster.hierarchy as shc
    plt.figure(figsize=(10,10))
    plt.title("Dendograms")
    dend = shc.dendrogram(shc.linkage(data,method='ward'))

def elbow(data,begin=1,end=5):
    distortions = []
    K = range(begin,end)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 
                                            'euclidean'), axis=1)) / data.shape[0])
    
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def KMean_build(data,k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    return kmeans

def K_Fold(model,X,Y,cv=5):
    from sklearn import model_selection
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=cv,random_state=42)
    rs = model_selection.cross_val_score(model,X,Y,cv=kfold)
    return rs
    
def SVM_classification(X,Y,kernel = 'rbf',C=1.0,degree = 3,gamma='scale'):
	from sklearn import svm
	if kernel=='poly':
		clf = svm.SVC(kernel='poly',C = C,degree = degree,gamma = gamma)
	else:
		clf = svm.SVC(kernel=kernel,C = C,gamma = gamma)
	clf.fit(X,Y)
	return clf

def SVM_Regression(X,Y,kernel = 'rbf',C=1.0,degree = 3,gamma='scale'):
	from sklearn import svm
	if kernel=='poly':
		clf = svm.SVR(kernel='poly',C = C,degree = degree,gamma = gamma)
	else:
		clf = svm.SVR(kernel=kernel,C = C,gamma = gamma)
	clf.fit(X,Y)
	return clf
	
def RandomForest_regression(X,Y,n_estimators=100,criterion = 'mse',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None):  	
  # criteria : mse/friedman_mse/mae
	# max_depth , min_samples_split , min_samples_leaf : int
	# max_features : auto/sqrt/log2 hoặc int/float
	# class_weight : dict/'balanced'
	from sklearn.ensemble import RandomForestRegressor
	clf = RandomForestRegressor(n_estimators=n_estimators,criterion = criterion,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_features = max_features)
	clf.fit(X,Y)
	return clf

def RandomForest_classification(X,Y,n_estimators=100,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,class_weight=None):
	# criteria : gini/entropy
	# splitter : best/random
	# max_depth , min_samples_split , min_samples_leaf : int
	# max_features : auto/sqrt/log2 hoặc int/float
	# class_weight : dict/'balanced'
	from sklearn.ensemble import  RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=n_estimators,criterion = criterion,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_features = max_features,class_weight = class_weight)
	clf.fit(X,Y)
	return clf
	
	
def DecisionTree_regression_model_build(X,Y,criterion = 'mse',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,class_weight=None):
  	# criteria : mse/friedman_mse/mae
	# splitter : best/random
	# max_depth , min_samples_split , min_samples_leaf : int
	# max_features : auto/sqrt/log2 hoặc int/float
	# class_weight : dict/'balanced'
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.metrics import accuracy_score
	model = DecisionTreeRegressor(criterion = criterion,splitter = splitter,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_features = max_features,class_weight = class_weight)
	model.fit(X, Y)
	return model

def DecisionTree_classify_model_build(X,Y,criterion = 'gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,class_weight=None):
	# criteria : gini/entropy
	# splitter : best/random
	# max_depth , min_samples_split , min_samples_leaf : int
	# max_features : auto/sqrt/log2 hoặc int/float
	# class_weight : dict/'balanced'
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.utils.validation import column_or_1d
	ds = DecisionTreeClassifier(criterion = criterion,splitter = splitter,max_depth = max_depth,min_samples_split = min_samples_split,min_samples_leaf = min_samples_leaf,max_features = max_features,class_weight = class_weight)
	ds.fit(X,Y)
	return ds
  

def KNN_regression_running_k(X_train,X_test,y_train,y_test,begin=2,end=11):
  from sklearn.neighbors import KNeighborsRegressor
  from sklearn.metrics import accuracy_score
  list_k = []
  list_score = []
  for K_value in range(begin,end):
    list_k.append(K_value)
    neigh = KNeighborsRegressor(n_neighbors = K_value)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    score = neigh.score(X_test, y_test)
    list_score.append(score)
    print("Accuracy is ", score,"% for K-Value:",K_value)

def KNN_classify_running_k(X_train,X_test,y_train,y_test):
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.metrics import accuracy_score
  list_k = []
  list_acc = []
  # thông thường thì chạy từ 2 đến căn bậc 2 của số lượng mẫu quan sát
  for K_value in range(2,int(y_train.shape[0]**0.5)):        
      list_k.append(K_value)
      neigh = KNeighborsClassifier(n_neighbors = K_value)
      neigh.fit(X_train, y_train) 
      y_pred = neigh.predict(X_test)
      acc = accuracy_score(y_test,y_pred)*100
      list_acc.append(acc)
      print("k = ", K_value,": Accuracy is ", accuracy_score(y_test,y_pred))
	  
	  
	  
def KNN_model_build(X,Y,K,kind="c"):
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.neighbors import KNeighborsRegressor
  if kind=="c":
    knn = KNeighborsClassifier(n_neighbors = K)
  else:
    knn = KNeighborsRegressor(n_neighbors = K)
  knn.fit(X,Y)
  return knn
  
def naive_bayes_model_build(X,Y,t="Gauss"):
  from sklearn.naive_bayes import GaussianNB
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.naive_bayes import BernoulliNB
  if t=="Gauss":
    nb = GaussianNB()
  elif t=="Multi":
    nb = MultinomialNB()
  else:
    nb = BernoulliNB()
  nb.fit(X,Y)
  return nb
  
def logistic_model_build(X,Y):
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression()
  clf.fit(X,Y)
  return clf
  
def linear_model_build(X,Y):
  from sklearn.linear_model import LinearRegression
  lm = LinearRegression()
  lm.fit(X,Y)
  Y_hat = lm.predict(X)
  return Y_hat,lm