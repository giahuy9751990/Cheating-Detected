import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pickle
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import ML_Model as md
import ML_Phantich as pt
import ML_Transform as mt
import warnings
warnings.filterwarnings('ignore')
from io import StringIO




file_ = "Model/onehotencoder_transformer_23112020.pkl"
onehot_encoder = pickle.load(open(file_, 'rb'))

file1_ = "Model/ada_random_forest_22112020.sav"
ada_md = pickle.load(open(file1_, 'rb'))


file2_ = "Model/onehotencoder_transformer_23112020.pkl"
onehot_encoder_2 = pickle.load(open(file2_, 'rb'))


file3_ = "Model/ada_random_forest__2_bonus_23112020.sav"
ada_md_2 = pickle.load(open(file3_, 'rb'))


data = pd.read_csv("Data Modeling/Data1.csv")
@st.cache
def read_dataframe_in():
	account = pd.read_csv("Data/Accounts Report SV1.csv",header = 1,sep=";",engine='python')
	order = pd.read_csv("Data/Orders Report SV1.csv",header = 1,sep=";",engine='python',error_bad_lines=False, warn_bad_lines=False)
	order_combines_1 = order[['Login','Reason','Time','Type','Symbol','Volume','Price','Swap','Profit','Comment']]
	account = account.sort_values(by="Login").iloc[2::]
	account['Login'] = account['Login'].astype(int)
	order_combines_1 = order_combines_1.merge(account,how="inner",left_on="Login",right_on="Login",validate="many_to_one")
	order_combines_1 = order_combines_1[['Login','Group','Reason', 'Time', 'Type', 'Symbol', 'Volume', 'Price', 'Swap','Balance','Credit','Reg.date',"Profit","Comment_x"]]
	order_combines_1.rename(columns={"Comment_x":"Comment"},inplace=True)


	account = pd.read_csv("Data/Accounts Report SV2.csv",header = 1,sep=";",engine='python')
	order = pd.read_csv("Data/Orders Report SV2.csv",header = 1,sep=";",engine='python',error_bad_lines=False, warn_bad_lines=False)
	order_combines_2 = order[['Login','Reason','Time','Type','Symbol','Volume','Price','Swap','Profit','Comment']]
	account = account.sort_values(by="Login").iloc[2::]
	account['Login'] = account['Login'].astype(int)
	order_combines_2 = order_combines_2.merge(account,how="inner",left_on="Login",right_on="Login",validate="many_to_one")
	order_combines_2 = order_combines_2[['Login','Group','Reason', 'Time', 'Type', 'Symbol', 'Volume', 'Price', 'Swap','Balance','Credit','Reg.date',"Profit","Comment_x"]]
	order_combines_2.rename(columns={"Comment_x":"Comment"},inplace=True)

	order_combines = pd.concat([order_combines_1,order_combines_2],axis=0)

	order_combines = order_combines[order_combines['Symbol'].map(lambda x: "fx" not in x)]
	order_combines = order_combines[order_combines['Credit']>0]
	order_combines['Reason'] = order_combines['Reason'].fillna("Mobile")
	order_combines = order_combines[order_combines['Symbol'].map(lambda x:x not in ['jpxjpy','usdhkd'])]

	order_combines.rename(columns={"Login":"Account","Credit":"Bonus"},inplace=True)

	df_transform = order_combines.copy()

	df_transform = df_transform.merge(df_transform.groupby(by=['Account'])['Type'].count().reset_index(),how="inner",left_on=["Account"],right_on=["Account"],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Type_x":"Type","Type_y":"Num_of_order"},inplace=True)

	df_transform.Time = pd.to_datetime(df_transform.Time)
	now = datetime.now()
	df_transform.Time = (now - df_transform.Time).dt.total_seconds()

	df_transform = df_transform.merge(df_transform.groupby(by=['Account',"Type"])['Time'].median().reset_index(),how="inner",left_on=["Account","Type"],right_on=["Account","Type"],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Time_x":"Time","Time_y":"Time_median"},inplace=True)


	df_transform = df_transform.merge(df_transform.groupby(by=['Account'])['Volume'].sum().reset_index(),how="inner",left_on=["Account"],right_on=["Account"],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Volume_x":"Volume","Volume_y":"Sum_of_vol"},inplace=True)

	df_transform = df_transform.merge(df_transform.groupby(by=['Account',"Type","Symbol"])['Volume'].sum().reset_index(),how="inner",left_on=['Account',"Type","Symbol"],right_on=['Account',"Type","Symbol"],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Volume_x":"Volume","Volume_y":"Sum_of_vol_type"},inplace=True)

	df_transform = df_transform.merge(df_transform.groupby(by=['Account'])['Symbol'].unique().map(lambda x: len(x)).reset_index(),how="inner",left_on=['Account'],right_on=['Account'],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Symbol_x":"Symbol","Symbol_y":"Num_of_sym"},inplace=True)

	df_transform = df_transform.merge(df_transform.groupby(by=['Account','Type'])['Volume'].median().reset_index(),how="inner",left_on=["Account",'Type'],right_on=["Account",'Type'],validate="many_to_one",copy=False)
	df_transform.rename(columns={"Volume_x":"Volume","Volume_y":"Volume_median"},inplace=True)

	df_transform['Bal_on_bonus'] = df_transform['Balance']/(df_transform['Bonus'] + 1)

	df_transform = df_transform[df_transform.Reason.map(lambda x: x in ['Client','Mobile'])]

	df_transform.reset_index(drop=True,inplace=True)

	X_predict = df_transform[['Volume_median','Symbol','Balance','Bonus','Num_of_order','Num_of_sym','Sum_of_vol','Sum_of_vol_type','Bal_on_bonus']]

	X_array = onehot_encoder_2.transform(X_predict).toarray()

	X_predict['predict'] = ada_md_2.predict(X_array)

	X_predict['Account'] = df_transform.Account
	X_predict['Type'] = df_transform.Type

	X_predict = X_predict[X_predict.predict==1]
	X_predict = X_predict.groupby(by=['Account','Type']).head(1)


	df_group = df_transform.copy()

	data_buy = df_group[df_group.Type == "buy"]
	data_sell = df_group[df_group.Type == "sell"]

	sum_data_buy = data_buy.groupby(by=['Account',"Symbol"])['Volume'].sum().reset_index()
	data_buy = data_buy.groupby(by=['Account']).head(1)
	data_buy.reset_index(drop=True,inplace=True)
	sum_data_buy = sum_data_buy.pivot(index='Account', columns='Symbol', values='Volume').fillna(0).reset_index()
	data_buy = data_buy.merge(sum_data_buy,how='inner',on="Account")

	sum_data_sell = data_sell.groupby(by=['Account',"Symbol"])['Volume'].sum().reset_index()
	data_sell = data_sell.groupby(by=['Account']).head(1)
	data_sell.reset_index(drop=True,inplace=True)
	sum_data_sell = sum_data_sell.pivot(index='Account', columns='Symbol', values='Volume').fillna(0).reset_index()
	data_sell = data_sell.merge(sum_data_sell,how='inner',on="Account")

	df_group = pd.concat([data_sell,data_buy]).fillna(0)

	df_group.reset_index(drop=True,inplace=True)

	scaler_column = df_group.drop(columns=['Sum_of_vol','Num_of_order','Account', 'Group', 'Reason', 'Time', 'Type', 'Symbol', 'Volume',
		   'Price', 'Swap','Reg.date','Comment',"Profit"]).columns

	scaler_minmax,df_scaler = mt.transform(df_group,scaler_column,scaler='minmax')

	df_scaler['Account'] = df_group.Account
	df_scaler['Num_of_order'] = df_group.Num_of_order
	df_scaler['Type'] = df_group.Type
	df_scaler['Sum_of_vol'] = df_group.Sum_of_vol
	df_scaler['Num_of_sym'] = df_group.Num_of_sym

	data_buy = df_scaler[df_scaler.Type == "buy"]
	data_sell = df_scaler[df_scaler.Type == "sell"]
	data_buy.reset_index(drop=True,inplace=True)
	data_sell.reset_index(drop=True,inplace=True)

	nbrs = NearestNeighbors(n_neighbors=2,p=2,algorithm="kd_tree")
	nbrs.fit(data_buy.drop(columns=['Account','Type','Num_of_order','Sum_of_vol','Num_of_sym','Sum_of_vol_type']))
	distances, indices = nbrs.kneighbors(data_sell.drop(columns=['Account','Type','Num_of_order','Sum_of_vol','Num_of_sym','Sum_of_vol_type']))
	indices_df = pd.DataFrame(indices,columns=['Account_0','Account_1'])
	distances_df = pd.DataFrame(distances,columns=['Distance_0','Distance_1'])
	data_sell['Account_1'] = data_buy.iloc[indices_df.Account_0].Account.values
	data_sell['Account_2'] = data_buy.iloc[indices_df.Account_1].Account.values
	data_sell['distance'] = distances_df.Distance_0

	nbrs = NearestNeighbors(n_neighbors=2,p=2,algorithm="kd_tree")
	nbrs.fit(data_sell.drop(columns=['Account', 'Type','Account_1','Account_2','distance','Num_of_order','Sum_of_vol','Num_of_sym','Sum_of_vol_type']))
	distances, indices = nbrs.kneighbors(data_buy.drop(columns=['Account','Type','Num_of_order','Sum_of_vol','Num_of_sym','Sum_of_vol_type']))
	indices_df = pd.DataFrame(indices,columns=['Account_0','Account_1'])
	distances_df = pd.DataFrame(distances,columns=['Distance_0','Distance_1'])
	data_buy['Account_1'] = data_sell.iloc[indices_df.Account_0].Account.values
	data_buy['Account_2'] = data_sell.iloc[indices_df.Account_1].Account.values
	data_buy['distance'] = distances_df.Distance_0

	df_group = pd.concat([data_sell,data_buy])
	df_group = df_group[['Num_of_sym','Num_of_order','Type','Sum_of_vol','Account','Account_1','Account_2','distance']]

	X_predict = X_predict.merge(df_group,how='inner',on=['Account','Type'])[['Account','Account_1','Account_2','Type','Balance','Bonus','Symbol','Num_of_order_x','Sum_of_vol_x']]
	X_predict.rename(columns={'Account_1':"Acc Gian Lan 1","Account_2":"Acc Gian Lan 2","Num_of_order_x":"So luong lenh","Sum_of_vol_x":"Tong So Volume"},inplace=True)
	return X_predict,df_group



X_predict,df_group=read_dataframe_in()


page = st.sidebar.selectbox("Choose a page",['Predict_data_1','Predict_data_2','Predict_all','Find_Nearest_Point','Account_Find'])


if page == 'Predict_data_1':
	st.header("This page use to predict a order with bonus 1")
	Volume_median = st.number_input('Volume Median : ',min_value=0.01)
	Balance = st.number_input('Balance : ',format="%i",min_value=0,value=1)
	Bonus = st.number_input('Bonus : ',format="%i",min_value=0,value=1)
	Number_of_order = st.number_input('Number Of Order : ',format="%i",min_value=1,value=1)
	symbol_box = st.selectbox("Chose Symbol Currency",data.Volume.unique())
	X_array = [[Volume_median,symbol_box,Balance,Bonus,Number_of_order,1,Volume_median*Number_of_order,Volume_median*Number_of_order,Balance/(Bonus+1)]]
	X_array = onehot_encoder.transform(X_array).toarray()
	st.write("Predict : ","Cheat" if ada_md.predict(X_array)==1 else "Normal")
elif page == 'Predict_data_2':
	st.header("This page use to predict a order with bonus 2")
	Volume_median = st.number_input('Volume Median : ',min_value=0.01)
	Balance = st.number_input('Balance : ',format="%i",min_value=0,value=1)
	Bonus = st.number_input('Bonus : ',format="%i",min_value=0,value=1)
	Number_of_order = st.number_input('Number Of Order : ',format="%i",min_value=1,value=1)
	symbol_box = st.selectbox("Chose Symbol Currency",data.Volume.unique())
	X_array = [[Volume_median,symbol_box,Balance,Bonus,Number_of_order,1,Volume_median*Number_of_order,Volume_median*Number_of_order,Balance/(Bonus+1)]]
	X_array = onehot_encoder_2.transform(X_array).toarray()
	st.write("Predict : ","Cheat" if ada_md_2.predict(X_array)==1 else "Normal")

elif page=="Predict_all":
	st.dataframe(X_predict,width=1000, height=800)

elif page == 'Account_Find':
	type_box_account = st.selectbox("Choose buy/sell order",['buy','sell'])
	account_input = st.number_input("Enter Account",format="%i",min_value=2)
	st.dataframe(df_group[(df_group.Account==account_input) & (df_group.Type==type_box_account)],width=1000, height=800)
else:
	type_box = st.selectbox("Choose buy/sell order",['buy','sell'])
	header = st.number_input("Choose header",format="%i",min_value=10)
	sum_of_vol = st.number_input("Choose number sum_of_vol",format="%i",min_value=2)
	num_of_order = st.number_input("Choose number num_of_order",format="%i",value=20,min_value=1,max_value=66)
	num_of_sym = st.number_input("Choose number num_of_sym",format="%i",value=6,min_value=1,max_value=6)
	st.header("This page use to find nearest other account")
	st.dataframe(df_group[(df_group.Type==type_box)&(df_group.Account != df_group.Account_1)&(df_group.Sum_of_vol>=sum_of_vol)&(df_group.Num_of_order<=num_of_order)&(df_group.Num_of_sym<=num_of_sym)].sort_values(by="distance",ascending=True).head(header),width=1000, height=800)