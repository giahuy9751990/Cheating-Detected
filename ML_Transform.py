import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing



def LabelEncoderProcess(df):
  from sklearn.preprocessing import LabelEncoder
  label_encoder = LabelEncoder()
  label_encoder.fit(df)
  return label_encoder.transform(df),label_encoder

def chia_bo_du_lieu(X,Y,test_size,random_state=42):
  from sklearn.model_selection import train_test_split
  X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=random_state)
  return X_train,X_test,Y_train,Y_test

def transform(df,feature_list,scaler="standard",threshold = None):
  if scaler=="log":
    return np.log(df[feature_list])
  elif scaler=="standard":
    scaler_tranform = preprocessing.StandardScaler()
  elif scaler=="minmax":
    scaler_tranform = preprocessing.MinMaxScaler()
  elif scaler=="robust":
    scaler_tranform = preprocessing.RobustScaler()
  elif scaler_tranform=="normalizer":
    scaler_tranform = preprocessing.Normalizer()
  elif scaler=="binarizer":
    scaler_tranform = preprocessing.Binarizer(threshold=threshold)
  
  scaler_transform = scaler_tranform.fit(df[feature_list])
  return scaler_transform,pd.DataFrame(scaler_transform.transform(df[feature_list]),columns=feature_list)

def tranform_polinominal(X,poli_degree):
    from sklearn.preprocessing import PolynomialFeatures
    pr = PolynomialFeatures(degree=poli_degree)
    pr.fit(X)
    return pr.transform(X),pr


