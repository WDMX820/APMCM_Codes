#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import warnings 
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error,mean_absolute_error
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100 
def score(y_true, y_pre):
    # MAPE
    print("MAPE :")
    print(mean_absolute_percentage_error(y_true, y_pre)) 
    # RMSE
    print("RMSE :")
    print(np.sqrt(mean_squared_error(y_true, y_pre))) 
    # MAE
    print("MAE :")
    print(mean_absolute_error(y_true, y_pre)) 
    # R2
    print("R2 :")
    print(np.abs(r2_score(y_true,y_pre)))

各国核武器数量的变化趋势。
# In[8]:


stockpiles=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='stockpiles')
stockpiles


# In[9]:


stockpiles[stockpiles['Stockpile']>0]['Country'].unique()


# In[10]:


i='United States'
temp1=stockpiles[stockpiles['Country']==i]
temp1=temp1.groupby('Year').sum()
temp1.reset_index(inplace=True,drop=False)
temp1


# In[11]:


import os 
save_path='./Q2/%s'%i
if os.path.exists(save_path):
    pass
else:
    os.mkdir(save_path)
    
temp1.to_excel('%s/temp1.xlsx'%(save_path),index=None)


# ### ------------------------------------------------------------------

# In[12]:


dataset=temp1['Stockpile']


# In[13]:


temp1


# In[14]:


import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model

# 将整型变为float
dataset = dataset.astype('float32')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.values.reshape(-1, 1))

 
def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX),numpy.array(dataY)


#训练数据太少 look_back并不能过大
look_back = 1
trainX,trainY  = create_dataset(dataset,look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# model.save(os.path.join("DATA","Test" + ".h5"))
# make predictions
 


# In[15]:


#模型验证
#model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)

#反归一化
trainPredict_ = scaler.inverse_transform(trainPredict)
trainY_ = scaler.inverse_transform(trainY)


# In[16]:


score(trainPredict_,trainY_)    


# In[17]:


plt.plot(temp1['Year'].values[1:],trainY_, label='observed data')
plt.plot(temp1['Year'].values[1:],trainPredict_, label='LSTM')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/stockpiles-year fitting diagram.jpg')
plt.show()


# In[18]:


x_input=trainY[-1]
predict_forword_number=101
predict_list=[]
predict_list.append(x_input)
while len(predict_list) < predict_forword_number:
    x_input = predict_list[-1].reshape((-1, 1, 1))
    yhat = model.predict(x_input, verbose=0)
    #预测新值
    predict_list.append(yhat)
    #取出    
    
bb=scaler.inverse_transform(np.array([ i.reshape(-1,1)[:,0].tolist() for i in predict_list]))[1:]
bb


# In[19]:


pd.DataFrame(bb).to_excel('%s/lstmpre.xlsx'%(save_path),index=None)


# ## -----------------------------------------------------------------

# In[20]:


dataset=pd.concat([temp1[['Year','Stockpile']],temp1['Stockpile'].shift(1)],axis=1)
dataset.dropna(inplace=True,axis=0)
dataset.columns=['Year','X','Y']


# In[21]:


dataset.to_excel('%s/dataset_.xlsx'%(save_path),index=None)


# In[22]:


import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')


train_X=dataset.iloc[:,1].values.reshape(-1, 1)
train_y=dataset.iloc[:,2].values.reshape(-1, 1)
test_X=dataset.iloc[::,1].values.reshape(-1, 1)
test_y=dataset.iloc[::,2].values.reshape(-1, 1)

#高维数据模型业界一般选择线性回归，速度快，准确率高
#线性回归
model_lr = LinearRegression()

model_lr.fit(train_X,train_y)
# print('线性回归')

print(score(model_lr.predict(test_X.reshape(-1, 1)),test_y))

plt.plot(temp1['Year'].values[1:],test_y, label='observed data')
plt.plot(temp1['Year'].values[1:],model_lr.predict(test_X.reshape(-1, 1)), label='LinearRegression')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/LinearRegression stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()


# In[23]:


aaaa=[]
for i in range(100):
    if i ==0:
        tt=model_lr.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lr.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0][0])
aaaa


# In[24]:


# Randomforest回归

model_rf = RandomForestRegressor()

model_rf.fit(train_X,train_y)
print('Randomforest回归')

print(score(model_rf.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')

plt.plot(temp1['Year'].values[1:],test_y, label='observed data')
plt.plot(temp1['Year'].values[1:],model_rf.predict(test_X.reshape(-1, 1)), label='Randomforest')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/Randomforest stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()
aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_rf.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_rf.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[25]:


# lgbm回归

model_lgb = lgb.LGBMRegressor()

model_lgb.fit(train_X,train_y)
print('lgbm回归')
print(score(model_lgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(temp1['Year'].values[:-1],test_y, label='observed data')
plt.plot(temp1['Year'].values[:-1],model_lgb.predict(test_X.reshape(-1, 1)), label='lgbm')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/lgbm stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_lgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[26]:


#  xgboost回归
model_xgb = xgb.XGBRegressor()

model_xgb.fit(train_X,train_y)
print('xgboost回归')
print(score(model_xgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(temp1['Year'].values[:-1],test_y, label='observed data')
plt.plot(temp1['Year'].values[:-1],model_xgb.predict(test_X.reshape(-1, 1)), label='XGBRegressor')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/XGBRegressor stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_xgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_xgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[27]:


temp1['rolling5']=temp1['Stockpile'].rolling(5).mean()


# In[28]:


a=[]
count=0
for i in temp1['Year']:
    count+=1
    if count%5==0:
        a.append(1)
    elif count==1:
        a.append(1)
    else:
        a.append(0)   


# In[29]:


a.reverse()
temp1['aa']=a
temp1


# In[30]:


tempp=temp1[temp1['aa']==1]
tempp.reset_index(inplace=True,drop=True)
dataset.dropna(inplace=True,axis=0)
tempp


# In[31]:


tempp.to_excel('%s/rolling5.xlsx'%(save_path),index=None)


# In[32]:


dataset=pd.concat([tempp[['Year','rolling5']],tempp['rolling5'].shift(1)],axis=1)
dataset.dropna(inplace=True,axis=0)
dataset.columns=['Year','X','Y']
dataset['Y']=dataset['Y'].astype('int')
dataset


# In[33]:


dataset.to_excel('%s/dataset_rolling5.xlsx'%(save_path),index=None)


# In[34]:


import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings 
warnings.filterwarnings('ignore')

train_X=dataset.iloc[:,1].values.reshape(-1, 1)
train_y=dataset.iloc[:,2].values.reshape(-1, 1)
test_X=dataset.iloc[::,1].values.reshape(-1, 1)
test_y=dataset.iloc[::,2].values.reshape(-1, 1)

#高维数据模型业界一般选择线性回归，速度快，准确率高
#线性回归
model_lr = LinearRegression()

model_lr.fit(train_X,train_y)
# print('线性回归')

print(score(model_lr.predict(test_X.reshape(-1, 1)),test_y))

plt.plot(dataset['Year'],test_y, label='observed data')
plt.plot(dataset['Year'],model_lr.predict(test_X.reshape(-1, 1)), label='LinearRegression')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()

plt.savefig('%s/LinearRegression stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()


# In[35]:


# Randomforest回归

model_rf = RandomForestRegressor()

model_rf.fit(train_X,train_y)
print('Randomforest回归')

print(score(model_rf.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')

plt.plot(dataset['Year'],test_y, label='observed data')
plt.plot(dataset['Year'],model_rf.predict(test_X.reshape(-1, 1)), label='Randomforest')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/Randomforest stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()
aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_rf.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_rf.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[36]:


# lgbm回归

model_lgb = lgb.LGBMRegressor()

model_lgb.fit(train_X,train_y)
print('lgbm回归')
print(score(model_lgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(dataset['Year'],test_y, label='observed data')
plt.plot(dataset['Year'],model_lgb.predict(test_X.reshape(-1, 1)), label='lgbm')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/lgbm stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_lgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[37]:


#  xgboost回归
model_xgb = xgb.XGBRegressor()

model_xgb.fit(train_X,train_y)
print('xgboost回归')
print(score(model_xgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(dataset['Year'],test_y, label='observed data')
plt.plot(dataset['Year'],model_xgb.predict(test_X.reshape(-1, 1)), label='XGBRegressor')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('%s/XGBRegressor stockpiles-year fitting diagram.jpg'%(save_path))
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_xgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_xgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[ ]:





# In[ ]:




