# In[1]:
import pandas as pd


# In[2]:
data_position=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='position')
data_position


# a)哪些国家曾经拥有过核武器？ 
# In[5]:
data_position[data_position['Status']==3]['Country'].unique()


# b)哪个国家在过去20年里核武器库存减少或增加幅度最大？ 
# In[6]:
data_stockpiles=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='stockpiles')
data_stockpiles


# In[7]:
temp=data_stockpiles[data_stockpiles['Year']>2002]
temp


# In[11]:
temp.to_excel('./ttt/Q1_2.xlsx',index=None)

# In[8]:

# 导入相关包
import matplotlib.pyplot as plt
import numpy as np

# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] 

# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False 

import warnings
warnings.filterwarnings('ignore')
# 魔法函数：在notebook嵌入图表
get_ipython().run_line_magic('matplotlib', 'inline')


# 定义一个画板，尺寸为(8,5)，dpi指定分辨率
plt.figure(figsize=(10, 10),dpi=90)



for i in temp['Country'].unique():
    ttemp=temp[temp['Country']==i]
    if ttemp.iloc[0,3]==ttemp.iloc[-1,3]:
        tt=0
    elif ttemp.iloc[0,3]>ttemp.iloc[-1,3]: 
        tt=-round((ttemp.iloc[0,3]-ttemp.iloc[-1,3])/ttemp.iloc[-1,3],ndigits=4)
    else:
        tt=round((ttemp.iloc[-1,3]-ttemp.iloc[0,3])/ttemp.iloc[-1,3],ndigits=4)
    print(i,tt)
    plt.plot(ttemp['Year'], ttemp['Stockpile'], linewidth=1.0, label=str(tt)+' '+i)

plt.title("nuclear weapons stockpiles in the last 20 years")
plt.xlabel("year")
plt.ylabel('Stockpile')

plt.legend(loc='best')
plt.savefig('./Q1/nuclear weapons stockpiles in the last 20 years.jpg')
plt.show()


# c)在哪五年中核武器试验最多？ 

# In[12]:


data_tests=pd.read_excel('2022_APMCM_E_Data.xlsx',sheet_name='tests')
data_tests


# In[13]:


temp=data_tests.groupby('Year').sum()
temp.sort_values(by='Tests',ascending=False).head(5)


# In[14]:


temp.to_excel('./ttt/Q1_3.xlsx',index=None)


# In[57]:


# 创建基础数据
n=5
x =[str(i) for i in list(temp.sort_values(by='Tests',ascending=False).head(n).index)]
y =list(temp.sort_values(by='Tests',ascending=False).head(n)['Tests'])


# 定义一个画板，尺寸为(5,5)，dpi指定分辨率
plt.figure(figsize=(10, 10),dpi=80)
plt.barh(x,y,align='center',color='steelblue',alpha=0.6)


# 为每个条形图添加数值标签
for x,y in enumerate(y):
    plt.text(y, x, f"{y}", ha='center')
plt.title("During which five years did nuclear weapon tests occur the most")
plt.xlabel("Tests")
plt.ylabel('year')
plt.savefig('./Q1/During which five years did nuclear weapon tests occur the most.jpg')
plt.show()


# d)在过去的10年里，哪个国家在核武器研究中最为活跃？ 

# In[ ]:


可以统计出各个国家在过去的10年里持有量的增幅、试验次数


# In[15]:


temp=data_stockpiles[data_stockpiles['Year']>2012]
temp


# In[16]:


aa=[]

plt.figure(figsize=(10, 10),dpi=90)


for i in temp['Country'].unique():
    ttemp=temp[temp['Country']==i]
    if ttemp.iloc[0,3]==ttemp.iloc[-1,3]:
        tt=0
    elif ttemp.iloc[0,3]>ttemp.iloc[-1,3]: 
        tt=-round((ttemp.iloc[0,3]-ttemp.iloc[-1,3])/ttemp.iloc[-1,3],ndigits=4)
    else:
        tt=round((ttemp.iloc[-1,3]-ttemp.iloc[0,3])/ttemp.iloc[-1,3],ndigits=4)
    print(i,tt)
    plt.plot(ttemp['Year'], ttemp['Stockpile'], linewidth=1.0, label=str(tt)+' '+i)
    aa.append([i,tt])
       
plt.title("Considering the increase or decrease in pursuit in the last 10 years")
plt.xlabel("year")
plt.ylabel('range')

plt.legend(loc='best')
plt.savefig('./Q1/Considering the increase or decrease in pursuit in the last 10 years.jpg')
plt.show() 


# In[107]:


aa


# In[17]:


temp=data_tests[data_tests['Year']>2012].groupby('Country').sum()
temp.sort_values(by='Tests',ascending=False)['Tests']


# In[18]:


temp.to_excel('./ttt/Q1_4.xlsx',index=None)


# In[116]:


# 创建基础数据
n=10
x =[str(i) for i in list(temp.sort_values(by='Tests',ascending=False).head(n).index)]
y =list(temp.sort_values(by='Tests',ascending=False).head(n)['Tests'])


# 定义一个画板，尺寸为(5,5)，dpi指定分辨率
plt.figure(figsize=(10, 10),dpi=80)
plt.barh(x,y,align='center',color='steelblue',alpha=0.6)


# 为每个条形图添加数值标签
for x,y in enumerate(y):
    plt.text(y, x, f"{y}", ha='center')
plt.title("During which 10 years did nuclear weapon tests occur the most")
plt.xlabel("Tests")
plt.ylabel('year')
plt.savefig('./Q1/During which 10 years did nuclear weapon tests occur the most.jpg')
plt.show()


# e)哪个国家从“不考虑使用核武器”到“拥有核武器”的过渡速度最快？ 

# In[117]:


data_position[data_position['Status']==3]['Country'].unique()


# In[133]:


# 创建基础数据
x=[]
y=[]
for i in data_position[data_position['Status']==3]['Country'].unique():
    temp=data_position[data_position['Country']==i]
    x.append(i)
    y.append(temp[temp['Status']==3].iloc[0,2]-temp[temp['Status']==0].iloc[-1,2])

# 定义一个画板，尺寸为(5,5)，dpi指定分辨率
plt.figure(figsize=(10, 10),dpi=80)
plt.barh(x,y,align='center',color='steelblue',alpha=0.6)


# 为每个条形图添加数值标签
for x,y in enumerate(y):
    plt.text(y, x, f"{y}", ha='center')
plt.title("transition year")
plt.xlabel("year")
plt.ylabel('Country')
plt.savefig('./Q1/transition year.jpg')
plt.show()



# In[143]:


temp=data_position[data_position['Country']=='South Africa']
temp.head(50)


# In[146]:


temp.tail(35)


# In[20]:


y


# In[150]:


x


# In[21]:


# 创建基础数据
x=[]
y=[]
for i in data_position[data_position['Status']==3]['Country'].unique():
    temp=data_position[data_position['Country']==i]
    x.append(i)
    y.append(temp[temp['Status']==3].iloc[0,2]-temp[temp['Status']==0].iloc[-1,2])
y[7]=1979-1968
# 定义一个画板，尺寸为(5,5)，dpi指定分辨率
plt.figure(figsize=(10, 10),dpi=80)
plt.barh(x,y,align='center',color='steelblue',alpha=0.6)
print(x)
print(y)
# 为每个条形图添加数值标签
for x,y in enumerate(y):
    plt.text(y, x, f"{y}", ha='center')
plt.title("transition year")
plt.xlabel("year")
plt.ylabel('Country')
plt.savefig('./Q1/transition year.jpg')
plt.show()



##a)哪些国家曾经拥有过核武器？ 
##直接统计出核弹持有量=3的国家即可
##
##b)哪个国家在过去20年里核武器库存减少或增加幅度最大？ 
##增幅/降幅＝（2022年期持有量-2002年持有量）／2002年持有量
##
##c)在哪五年中核武器试验最多？ 
##按时间分组聚合统计核武器试验次数，降序排序，输出最多试验的5年
##
##d)在过去的10年里，哪个国家在核武器研究中最为活跃？ 
##可以统计出各个国家在过去的10年里从考虑到追求的增幅、从追求到占有的增幅、持有量的增幅、试验次数，用综合评价法，如topsis、rsr秩比法等等计算出综合活跃排序
##
##e)哪个国家从“不考虑使用核武器”到“拥有核武器”的过渡速度最快？
##考虑从0到3最快的国家
