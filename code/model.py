import time

import pandas as pd 

import numpy as np

from pandas import read_csv

import xgboost as xgb

import csv

from sklearn.model_selection import train_test_split

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn import  linear_model

import statsmodels

from sklearn.neural_network import MLPRegressor  

from sklearn.preprocessing import StandardScaler

#导入数据
train = read_csv('t15.csv', header=0,encoding = "gbk")


train_data = train.values

sku = train['sku_ID'].values

#定义R-square函数
def RSquare(predict,real):
    average = np.sum(real)/real.shape[0]
    predictb = np.zeros((predict.shape[0]))
    realb = np.zeros((predict.shape[0]))
    resi = np.zeros((predict.shape[0]))
    for i in range(predict.shape[0]):
        predictb[i] = (predict[i]-average)**2
        realb[i] = (real[i]-average)**2
        resi[i] = (real[i]-predict[i])**2
    r1_square = np.sum(predictb)/np.sum(realb)
    r2_square = 1-np.sum(resi)/np.sum(realb)
    print(r1_square)
    print(r2_square)

#定义RMSE函数 
def RMSE(predictions, targets):
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    print("The RMSE of the test set is %s:" %rmse)
    return rmse

    
#将预测值小于0的转为0 
def adjust(predict):
    final=np.zeros((predict.shape[0]))
    for i in range(predict.shape[0]):
        if predict[i] >=0:
            final[i] = predict[i]
        else:
            final[i] = 0
    return final

#可以将真实值较小的去掉，用以可视化分析
def dropsmall(real,ypred1,ypred2,ypred3,N):
    i = 0 
    while i <= real.shape[0]:
        if real[i] < N:
            real = np.delete(real,i)
            ypred1 = np.delete(ypred1,i)
            ypred2 = np.delete(ypred2,i)
            ypred3 = np.delete(ypred3,i)
            i = i 
        else:
            i = i + 1
        try:
            a = real[i]
        except:
            print('error')
            break

    return real,ypred1,ypred2,ypred3
    
    
#在原始数据中划分训练集和测试集(计算均值)


data_x = train_data[:,3:32]

X = data_x

y = train_data[:,32]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1323, train_size=0.95)



#在原始数据中划分训练集和测试集（计算方差）


data_x = train_data[:,3:32]

X = data_x

y = train_data[:,33]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1323, train_size=0.85)


#linear model
lin = linear_model.LinearRegression(fit_intercept=True)

lin.fit(x_train, y_train)

y_pred_lin = lin.predict(x_test) 

y_pred_lin = adjust(y_pred_lin) 

y_pred_lin = y_pred_lin.reshape(y_pred_lin.shape[0],1)

RMSE(y_pred_lin,y_test)

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test, color = 'black', label = 'real sales std')
ax.plot(y_pred_lin, color = 'green', label = 'predicted the std of sales')
ax.legend(); ax.grid()
modelname = 'linear regression'
ax.set_title(modelname, fontsize=14)

plt.show()

#ANN
clf = MLPRegressor(solver='lbfgs', alpha=1e-3,hidden_layer_sizes=(3,4,5,8,4), random_state=123,max_iter=1000)
                   
clf.fit(x_train, y_train)

y_pred_mlp = clf.predict(x_test) 

y_pred_mlp = adjust(y_pred_mlp) 

y_pred_mlp = y_pred_mlp.reshape(y_pred_mlp.shape[0],1)

#RSquare(y_pred,y_test)
#RSquare(y_hat,y_train)
RMSE(y_pred_mlp,y_test)

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test, color = 'black', label = 'real sales std')
ax.plot(y_pred_mlp, color = 'green', label = 'predicted the std of sales')
ax.legend(); ax.grid()
modelname = 'BP'
ax.set_title(modelname, fontsize=14)

plt.show()

#XGBoost
dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_test)
watchlist = [(dtrain,'train')]

params={'booster':'gblinear',
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'early_stopping_rounds':100,
    'max_depth':200,
    'lambda':0,
    'gamma':0.2,
    'subsample':0.9,
    'colsample_bytree':0.95,
    'min_child_weight':1,
    'eta': 0.01,
    'seed':0,
    'nthread':-1,
     'silent':1}
     
bst = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist) 

ypred_xg = bst.predict(dtest)

ypred_xg = adjust(ypred_xg) 

y_pred_xg = ypred_xg.reshape(ypred_xg.shape[0],1)

#RSquare(ypred_xg,y_test)
#RSquare(y_hat_xg,y_train)
RMSE(y_pred_xg,y_test)

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test, color = 'black', label = 'real sales std')
ax.plot(y_pred_xg, color = 'green', label = 'predicted the std of sales')
ax.legend(); ax.grid()
modelname = 'XGBoost'
ax.set_title(modelname, fontsize=14)


RMSE(y_pred_xg,y_test)

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test, color = 'black', label = 'real sales std')
ax.plot(y_pred_xg, color = 'green', label = 'predicted the std of sales')
ax.legend(); ax.grid()
modelname = 'XGBoost'
ax.set_title(modelname, fontsize=14)

plt.show()


#同时展示三幅图
fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test, color = 'black', label = 'Real std ')
ax.plot(y_pred_lin, color = 'blue', label = 'predicted std in linear model')
ax.plot(y_pred_mlp, color = 'red', label = 'predicted std in MLP model')
ax.plot(y_pred_xg, color = 'green', label = 'predicted std in XGBoost model')
ax.legend(); ax.grid()
modelname = 'Comparing with different model'
ax.set_title(modelname, fontsize=14)

plt.show()



#去掉原本值比较小的再展示
y_test_a,y_pred_lin_a,y_pred_mlp_a,y_pred_xg_a = dropsmall(y_test,y_pred_lin,y_pred_mlp,y_pred_xg,20)

fig, ax = plt.subplots(figsize=(14,4))
ax.plot(y_test_a, color = 'black', label = 'Real mean')
ax.plot(y_pred_lin_a, color = 'blue', label = 'predicted std in linear model')
ax.plot(y_pred_mlp_a, color = 'red', label = 'predicted std in MLP model')
ax.plot(y_pred_xg_a, color = 'green', label = 'predicted std in XGBoost model')
ax.legend(); ax.grid()
modelname = 'Comparing'
ax.set_title(modelname, fontsize=14)

plt.show()


#导入数据预测
test_x = train_data[:,3:32]
test_x=xgb.DMatrix(test_x)
y_std = train_data[:,33]

y_pred_mlp_std = bst.predict(test_x).astype(int) 

y_pred_mlp_std = adjust(y_pred_mlp_std) 

y_pred_mlp_std = y_pred_mlp_std.reshape(y_pred_mlp_std.shape[0],1).astype(int)



sku=pd.DataFrame(sku)
sku.columns=['sku_ID']

sku['predict21-30u']=y_pred_mlp
sku['predict21-30std']=y_pred_mlp_std 
sku.to_csv('predictu.csv',encoding='gbk',index=True)

pj.to_csv('predict21-30.csv',encoding='gbk',index=True)