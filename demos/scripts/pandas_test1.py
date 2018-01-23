import numpy as np, pandas as pd

arr1 = np.arange(10)
s1 = pd.Series(arr1)

dic1 = {'a':10,'b':20,'c':30,'d':40,'e':50}
s2 = pd.Series(dic1)

arr2 = np.array(np.arange(12)).reshape(4,3)
df1 = pd.DataFrame(arr2)

dic2 = {'a':[1,2,3,4],'b':[5,6,7,8],
        'c':[9,10,11,12],'d':[13,14,15,16]}
df2 = pd.DataFrame(dic2)
dic3 = {'one':{'a':1,'b':2,'c':3,'d':4},
        'two':{'a':5,'b':6,'c':7,'d':8},
        'three':{'a':9,'b':10,'c':11,'d':12}}
df3 = pd.DataFrame(dic3)

df4 = df3[['one','three']]
s3 = df3['one']

s4 = pd.Series(np.array([1,1,2,3,5,8]))
s4.index = ['a','b','c','d','e','f']
# s4[3]
# s4['e']
# s4[[1,3,5]]
# s4[['a','b','d','f']]
# s4[:4]
# s4['c':]
# s4['b':'e']

s5 = pd.Series(np.array([10,15,20,30,55,80]),
               index = ['a','b','c','d','e','f'])
s6 = pd.Series(np.array([12,11,13,15,14,16]),
               index = ['a','c','g','b','d','f'])
s5 + s6
s5/s6

np.random.seed(1234)
d1 = pd.Series(2*np.random.normal(size = 100)+3)
d2 = np.random.f(2,4,size = 100)
d3 = np.random.randint(1,100,size = 100)
d1.count()  #非空元素计算
d1.min()    #最小值
d1.max()    #最大值
d1.idxmin() #最小值的位置，类似于R中的which.min函数
d1.idxmax() #最大值的位置，类似于R中的which.max函数
d1.quantile(0.1)    #10%分位数
d1.sum()    #求和
d1.mean()   #均值
d1.median() #中位数
d1.mode()   #众数
d1.var()    #方差
d1.std()    #标准差
d1.mad()    #平均绝对偏差
d1.skew()   #偏度
d1.kurt()   #峰度
d1.describe()   #一次性输出多个描述性统计指标

def stats(x):
    return pd.Series([x.count(),x.min(),x.idxmin(),
               x.quantile(.25),x.median(),
               x.quantile(.75),x.mean(),
               x.max(),x.idxmax(),
               x.mad(),x.var(),
               x.std(),x.skew(),x.kurt()],
              index = ['Count','Min','Whicn_Min',
                       'Q1','Median','Q3','Mean',
                       'Max','Which_Max','Mad',
                       'Var','Std','Skew','Kurt'])
stats(d1)

df = pd.DataFrame(np.array([d1,d2,d3]).T,columns=['x1','x2','x3'])
df.head()
df.apply(stats)


from pandas import Series,DataFrame
a=[['刘玄德','男','语文',98.],['刘玄德','男','体育',60.],['关云长','男','数学',60.],['关云长','男','语文',100.]]
af=DataFrame(a,columns=['name','sex','course','score'])
af.set_index(['name','sex','course'],inplace=True)
t1=af.unstack(level=2)
t2=t1.mean(axis=1,skipna=True)
t1['平均分']=t2
t1.fillna(0)


import numpy as np
import pandas as pd
df = pd.DataFrame(np.random.randint(0, 100, 100), columns=['score'])
# 以所在区间作为标签。如 x=5，返回：'[0-10]'
def make_label(x, step=10):
    m = x // step
    return '[{}-{}]'.format(m * step, (m + 1) * step)
# df['level'] = df['score'].map(make_label)
df['level'] = df['score'].map(lambda x: make_label(x, step=10))  # 改变区间长度为15
res = df.groupby('level').size()
print(df.head())
print(res)


# 遍历

for index,row in df.iterrows():
    for a in row:
        print (a)
    print (index) #获取行的索引
    print (row.a) #根据列名获取字段
    print (row[0])#根据列的序号（从0开始）获取字段


import pandas as pd
dict=[[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[4,5,6,7,8,9],[5,6,7,8,9,10]]
data=pd.DataFrame(dict)
print(data)
for indexs in data.index:
    print(data.loc[indexs].values[0:-1])