#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd

#a)根据所附数据或您收集的数据，建立数学模型，预测核武器数量以及预测未来100年拥有核武器的国家； 
#b)预测未来100年核武器数量的变化趋势、2123年核武器总数和各国核武器数量的变化趋势。
# In[25]:


proliferation=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='proliferation')
proliferation


# In[ ]:





# In[3]:


a=[]
count=0
for i in proliferation['Year']:
    count+=1
    if count%10==0:
        a.append(1)
    elif count==1:
        a.append(1)
    else:
        a.append(0)       


# In[4]:


a.reverse()
proliferation['huizong']=a


# In[5]:


temp=proliferation[proliferation['huizong']==1]
temp.to_excel('./Q2/Possession.xlsx',index=None)
temp


# In[6]:


stockpiles=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='stockpiles')
stockpiles


# In[33]:


stockpiles[stockpiles['Year']==2022].sort_values(by='Stockpile',ascending=False)


# In[7]:


temp1=stockpiles.groupby('Year').sum()
temp1.reset_index(inplace=True,drop=False)
temp1


# In[8]:


a=[]
count=0
for i in temp1['Year']:
    count+=1
    if count%10==0:
        a.append(1)
    elif count==1:
        a.append(1)
    else:
        a.append(0)   
        
a.reverse()
temp1['huizong']=a

temp1.to_excel('./Q2/stockpiles1.xlsx',index=None)
temp1[temp1['huizong']==1].to_excel('./Q2/stockpiles10.xlsx',index=None)
temp1


# In[9]:


# 导入相关包
import matplotlib.pyplot as plt
import numpy as np

# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] 

# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False 

# 魔法函数：在notebook嵌入图表
get_ipython().run_line_magic('matplotlib', 'inline')

# 定义一个画板，尺寸为(8,5)，dpi指定分辨率
plt.figure(figsize=(20, 10),dpi=90)



plt.plot(temp1['Year'], temp1['Stockpile'])

plt.title("stockpiles-year diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')

plt.savefig('./Q2/stockpiles-year diagram.jpg')
plt.show()


# In[10]:


import pandas as pd
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


# ### ------------------------------------------------------------------

# In[41]:


import sys
get_ipython().system('{sys.executable} -m  pip install --upgrade pip')
get_ipython().system('{sys.executable} -m pip install tensorflow  -i https://pypi.tuna.tsinghua.edu.cn/simple')
get_ipython().system('{sys.executable} -m pip install keras')


# In[13]:





# In[14]:


dataset=temp1[temp1['huizong']==1]['Stockpile']
dataset


# In[32]:


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
 


# In[33]:


#模型验证
#model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)

#反归一化
trainPredict_ = scaler.inverse_transform(trainPredict)
trainY_ = scaler.inverse_transform(trainY)


# In[34]:


score(trainPredict_,trainY_)    


# In[15]:


plt.plot(temp1['Year'].values[1:],trainY_, label='observed data')
plt.plot(temp1['Year'].values[1:],trainPredict_, label='LSTM')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/stockpiles-year fitting diagram.jpg')
plt.show()


# In[35]:


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


# In[36]:


pd.DataFrame(bb).to_excel('lstmpre.xlsx',index=None)


# ## -----------------------------------------------------------------

# In[63]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple')
get_ipython().system('{sys.executable} -m pip install lightgbm -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[39]:


dataset=pd.concat([temp1[['Year','Stockpile']],temp1['Stockpile'].shift(1)],axis=1)
dataset.dropna(inplace=True,axis=0)
dataset.columns=['Year','X','Y']


# In[40]:


dataset


# In[41]:


dataset.to_excel('./Q2/dataset_.xlsx',index=None)


# In[42]:


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
plt.savefig('./Q2/LinearRegression stockpiles-year fitting diagram.jpg')
plt.show()


# In[43]:


aaaa=[]
for i in range(100):
    if i ==0:
        tt=model_lr.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lr.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0][0])
aaaa


# In[90]:


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
plt.savefig('./Q2/Randomforest stockpiles-year fitting diagram.jpg')
plt.show()
aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_rf.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_rf.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[97]:


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
plt.savefig('./Q2/lgbm stockpiles-year fitting diagram.jpg')
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_lgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[98]:


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
plt.savefig('./Q2/XGBRegressor stockpiles-year fitting diagram.jpg')
plt.show()

aaaa=[]
for i in range(10):
    if i ==0:
        tt=model_xgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_xgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[15]:


temp1['rolling5']=temp1['Stockpile'].rolling(5).mean()


# In[16]:


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


# In[17]:


a.reverse()
temp1['aa']=a
temp1


# In[18]:


tempp=temp1[temp1['aa']==1]
tempp.reset_index(inplace=True,drop=True)
tempp.dropna(inplace=True)
tempp


# In[49]:


tempp.to_excel('./Q2/rolling5.xlsx',index=None)


# In[19]:


dataset=pd.concat([tempp[['Year','rolling5']],tempp['rolling5'].shift(1)],axis=1)
dataset.dropna(inplace=True,axis=0)
dataset.columns=['Year','X','Y']
dataset['Y']=dataset['Y'].astype('int')
dataset


# In[51]:


dataset.to_excel('./Q2/dataset_rolling5.xlsx',index=None)


# In[57]:


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

plt.plot(tempp['Year'].values[1:],test_y, label='observed data')
plt.plot(tempp['Year'].values[1:],model_lr.predict(test_X.reshape(-1, 1)), label='LSTM')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/LinearRegression stockpiles-year-rolling5 fitting diagram.jpg')
plt.show()



aaaa=[]
for i in range(20):
    if i ==0:
        tt=model_lr.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lr.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    print(tt[0][0])
    aaaa.append(tt[0][0])

    
    
ttt=pd.DataFrame(tempp['Year'].values[1:])
ttt['observed']=test_y
ttt['LSTM']=model_lr.predict(test_X.reshape(-1, 1))

DD=pd.DataFrame()
DD['LSTM_pre']=aaaa


# In[37]:


ttt=pd.DataFrame(tempp['Year'].values[1:])
ttt['observed']=test_y
ttt['LSTM']=model_lr.predict(test_X.reshape(-1, 1))

DD=pd.DataFrame()
DD['LSTM_pre']=aaaa


# In[38]:


DD


# In[58]:


# Randomforest回归

model_rf = RandomForestRegressor()

model_rf.fit(train_X,train_y)
print('Randomforest回归')

print(score(model_rf.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')

plt.plot(tempp['Year'].values[1:],test_y, label='observed data')
plt.plot(tempp['Year'].values[1:],model_rf.predict(test_X.reshape(-1, 1)), label='LGBM')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/Randomforest stockpiles-year-rolling5 fitting diagram.jpg')
plt.show()
aaaa=[]
for i in range(20):
    if i ==0:
        tt=model_rf.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_rf.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa

ttt['LGBM']=model_rf.predict(test_X.reshape(-1, 1))


DD['LGBM_pre']=aaaa


# In[ ]:


for i in range(20):
    if i ==0:
        tt=model_rf.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_rf.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0][0])
aaaa

ttt['LGBM']=model_rf.predict(test_X.reshape(-1, 1))

DD=pd.DataFrame()
DD['LGBM_pre']=aaaa


# In[48]:


DD


# In[57]:


# lgbm回归

model_lgb = lgb.LGBMRegressor()

model_lgb.fit(train_X,train_y)
print('lgbm回归')
print(score(model_lgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(tempp['Year'].values[:-1],test_y, label='observed data')
plt.plot(tempp['Year'].values[:-1],model_lgb.predict(test_X.reshape(-1, 1)), label='lgbm')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/lgbm stockpiles-year-rolling5 fitting diagram.jpg')
plt.show()

aaaa=[]
for i in range(20):
    if i ==0:
        tt=model_lgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_lgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa


# In[63]:


#  xgboost回归
model_xgb = xgb.XGBRegressor()

model_xgb.fit(train_X,train_y)
print('xgboost回归')
print(score(model_xgb.predict(test_X.reshape(-1, 1)),test_y))

print('\n-----------')
plt.plot(tempp['Year'].values[:-1],test_y, label='observed data')
plt.plot(tempp['Year'].values[:-1],model_xgb.predict(test_X.reshape(-1, 1)), label='LGBM-PSO')
plt.title("stockpiles-year fitting diagram ")
plt.xlabel("Year")
plt.ylabel('Stockpile')
plt.legend()
plt.savefig('./Q2/XGBRegressor stockpiles-year-rolling5 fitting diagram.jpg')
plt.show()

aaaa=[]
for i in range(20):
    if i ==0:
        tt=model_xgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_xgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa

aaaa=[]
for i in range(20):
    if i ==0:
        tt=model_xgb.predict(test_X[-1].reshape(-1, 1))
    else:
        tt=model_xgb.predict(np.array([aaaa[-1]]).reshape(-1, 1))
    aaaa.append(tt[0])
aaaa

ttt['LGBM-PSO']=model_xgb.predict(test_X.reshape(-1, 1))


DD['LGBM-PSO_pre']=aaaa


# In[65]:


ttt.to_excel('./ttt/各模型预测数据与真实数据.xlsx',index=None)


# In[67]:


DD.to_excel('./ttt/各模型对未来100年核弹数量预测结果.xlsx',index=None)

