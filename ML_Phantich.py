import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing



def dem_null(df):
    l = len(df.iloc[:,0])
    print("Tổng cộng có ",l," samples.")
    [print("- Biến ",i," : null ",df[i].isnull().sum(),"/",l,", chiếm ",df[i].isnull().sum()/l*100,"%.") for i in df.columns]

def phan_tich_don_bien_continous(df,lst):
    for i in lst:
        print("--------------------- Phân Tích đơn biến :", i , "-----------------------")
        print(df[i].describe())
        print("Median :" ,df[i].median())
        print("Mode :" ,df[i].mode())
        print("Skew ",df[i].skew())
        print("Kurtosis",df[i].kurtosis())
        sb.boxplot(df[i])
        plt.title(i)
        plt.show()
        sb.distplot(df[i])
        plt.title(i)
        plt.show()

def phan_tich_don_bien_categorical(df,lst):
    for i in lst:
        print("----------------- Phân tích biến", i ,"--------------------")
        count = df[i].value_counts()
        plt.bar(count.index,count.values)
        plt.title(count.name)
        plt.show()

def phan_tich_2_bien_lien_tuc(df,lst):
    print(df[lst].corr())
    g = sb.PairGrid(df[lst])
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter);


def phan_tich_2_bien_category(df,var_1,var_2):
    from scipy.stats import chi2_contingency
    from scipy.stats import chi2
    print("----------- Phân tích 2 biến " , var_1 ,"và ",var_2  )
    table = pd.crosstab(df[var_1],df[var_2])
    print(table)
    table.plot(kind='bar',stacked=True)
    stat,p,dof,expected = chi2_contingency(table)
    prob=0.95
    critical = chi2.ppf(prob,dof)
    alpha = 1.0 - prob
    print("significance=%.3f,p=%.3f"%(alpha,p))
    if p <= alpha:
        print("Dependent")
    else:
        print("Independent")

def phan_tich_bien_lien_tuc_va_category(df,var_1,var_2):
    # phân tích đến 2 biến
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    d_melt=df[[var_1,var_2]]
    plt.figure(figsize=(12,10))
    sb.boxplot(x=var_1,y=var_2,data=d_melt,palette="Set3")
    plt.show()
    st = var_2 + "~ C(" + var_1 + ")"
    model =ols(st,data=d_melt).fit()
    anova_table = sm.stats.anova_lm(model,typ=2)
    print(anova_table)

def liet_ke_cot(df,st):
  return [i for i in df.columns if (df[i].dtype).name==st]

def xem_cot_numeric(df,lst):
  j=1
  for i in lst:
    print(j, ": ", i , " Len=",len(df[i]), "Giá Trị :", df[i].unique() if len(df[i].unique())<150 else "")
    j = j+1
    
def xem_cot_category(df,lst):
  j=1
  for i in lst:
    print(j, ": ", i , " Len=",len(df[i]), "Giá Trị :", df[i].unique())
    j = j+1
    
def xem_ngoai_le(df,var_continous,thump_up_number = 1.5):
  q_25,q_50,q_75 = df[var_continous].quantile([.25,.5,.75])
  IQR = q_75-q_25
  min_value = df[var_continous].min()
  max_value = df[var_continous].max()
  IQR_range = thump_up_number*IQR
  lower = q_25-IQR_range
  upper = q_75+IQR_range
  df_now = df[(df[var_continous]<lower) | (df[var_continous]>upper)]
  df_index = df_now.index
  return lower,upper,df_now,df_index


def so_sanh_df(df1,df2,var_columns):
  print("--------- Chưa Loại ------------")
  print(df1[var_columns].describe())
  print("--------- Đã Loại ------------")
  print(df2[var_columns].describe())
  
  


def loc_correlation(df,lst,threshold=0.7):
  pd.options.display.max_columns = 500
  print("Threshold : ",threshold)
  print(df[lst].corr()[df[lst].corr()>threshold])
  
  
def Feature_Selection_For_Classify_continous_Category(X,Y):
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_classif
  print("- Sử dụng SelectKBest với ANOVA")
  bestfeatures = SelectKBest(score_func=f_classif,k='all')
  fit = bestfeatures.fit(X,Y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']
  print(featureScores)

def Feature_Selection_For_Classify_Cateogry_Category(X,Y):
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2
  print("- Sử dụng SelectKBest với chi2")
  bestfeatures = SelectKBest(score_func=chi2,k='all')
  fit = bestfeatures.fit(X,Y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']
  print(featureScores)


def Feature_Selection_For_Regression(X,Y):
  # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
  # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_regression
  from sklearn.ensemble import ExtraTreesRegressor
  print("- Sử dụng SelectKBest")
  bestfeatures = SelectKBest(score_func=f_regression,k='all')
  fit = bestfeatures.fit(X,Y)
  dfscores = pd.DataFrame(fit.scores_)
  dfcolumns = pd.DataFrame(X.columns)
  featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  featureScores.columns = ['Specs','Score']
  print(featureScores)
  print("- Sử dụng ExtraTreesRegressor")
  model = ExtraTreesRegressor()
  model.fit(X,Y)
  dfextra = pd.DataFrame(data=model.feature_importances_,index=X.columns,columns=['Specs'])
  print(dfextra)


def find_best_polinomial(X_train,X_test,Y_train,Y_test):
  import numpy as np
  import matplotlib.pyplot as plt 

  from sklearn.linear_model import LinearRegression
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import train_test_split

  rmses = []
  degrees = np.arange(1, 10)
  min_rmse, min_deg = 1e10, 0
  for deg in degrees:

      # Train features
      poly_features = PolynomialFeatures(degree=deg, include_bias=False)
      x_poly_train = poly_features.fit_transform(X_train)

      # Linear regression
      poly_reg = LinearRegression()
      poly_reg.fit(x_poly_train, Y_train)

      # Compare with test data
      x_poly_test = poly_features.fit_transform(X_test)
      poly_predict = poly_reg.predict(x_poly_test)
      poly_mse = mean_squared_error(Y_test, poly_predict)
      poly_rmse = np.sqrt(poly_mse)
      rmses.append(poly_rmse)

      # Cross-validation of degree
      if min_rmse > poly_rmse:
          min_rmse = poly_rmse
          min_deg = deg

  # Plot and present results
  print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))

  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot(111)
  ax.plot(degrees, rmses)
  ax.set_yscale('log')
  ax.set_xlabel('Degree')
  ax.set_ylabel('RMSE')
  
  
  
  
def danh_gia_mo_hinh_logistic(Y,Y_hat):
  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix
  print("- Precision and Recall Metric")
  print(classification_report(Y, Y_hat))
  print("- Confusion Metric")
  print(confusion_matrix(Y, Y_hat))

def danh_gia_mo_hinh_linear(Y,Y_hat):
  import seaborn as sns
  import matplotlib.pyplot as plt
  from sklearn.metrics import mean_squared_error
  print("MSE:",mean_squared_error(Y, Y_hat))
  plt.figure(figsize=(10,10))
  ax1 = sns.distplot(Y,hist=False,color='r',label="Actual")
  sns.distplot(Y_hat,hist=False,color='b',label="Fitted Values",ax=ax1)
  plt.title("Actual and Fitted Values")
  plt.show()
  plt.close()


def ve_roc_curve(Y,Y_hat):
  from sklearn.metrics import roc_curve
  import matplotlib.pyplot as plt
  fpr,tpr,thresholds = roc_curve(Y,Y_hat)
  plt.plot([0,1],[0,1],linestyle='--')
  plt.plot(fpr,tpr,marker='.')
  plt.show()
  
def ve_cay_decision(model,list_columns,list_class_name):
	# grapg.write_pdf("sdddd.pdf") để lưu lại trong file pdf
	# https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/
	# http://viz-js.com/ nơi copy file txt và in ra cây nếu cần
	# xuất file txt
	# with open(path + "practice/Chapter6_Decision_Tree/iris_1505.txt", "w") as f:
	# f = tree.export_graphviz(clf, out_file=f,feature_names=X.columns,class_names=iris.iris)
	from sklearn import tree
	import pydotplus
	from IPython.display import Image
	dot_data = tree.export_graphviz(model,out_file=None,feature_names = list_columns,class_names=list_class_name)
	graph = pydotplus.graph_from_dot_data(dot_data)
	Image(graph.create_png())
	return graph
