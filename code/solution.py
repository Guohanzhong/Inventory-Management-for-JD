#!/usr/bin/env python
# coding: utf-8

# 部分因子的构造代码
import pandas as pd
import numpy as np

#设置提取的数据的起止日期（前20和20天的）
open_day='2018-03-11'
close_day='2018-03-31'

# 分别取click表、delivery表、order表在日期内的数据
click=pd.DataFrame(pd.read_csv('click.csv',encoding='utf-8-sig',))
click['request_time']=click['request_time'].astype('datetime64')#将数据转为日期的格式
con1=click['request_time']>=open_day
con2=click['request_time']<close_day
click=click[con1&con2]

delivery=pd.DataFrame(pd.read_csv('delivery.csv',encoding='utf-8-sig',))
delivery['ship_out_time']=delivery['ship_out_time'].astype('datetime64')
con1=delivery['ship_out_time']>=open_day
con2=delivery['ship_out_time']<close_day
delivery=delivery[con1&con2]

order=pd.DataFrame(pd.read_csv('order.csv',encoding='utf-8-sig',))
order['order_time']=order['order_time'].astype('datetime64')
con1=order['order_time']>=open_day
con2=order['order_time']<close_day
order=order[con1&con2]

# 16 转化率
#点击数
df_click=pd.pivot_table(click,index=['sku_ID'],values=['request_time'],aggfunc=[len])
#订单数
df_order=pd.pivot_table(order,index=['sku_ID'],values=['order_time'],aggfunc=[len])

df_click.columns =[s1 +'_'+ str(s2) for (s1,s2) in df_click.columns.tolist()]#转化每一列的名字，下同
df_click.reset_index(inplace=True)
df_order.columns =[s1 +'_'+ str(s2) for (s1,s2) in df_order.columns.tolist()]
df_order.reset_index(inplace=True)

df_merge=pd.merge(df_click,df_order,how='left')
df_merge=df_merge.fillna(0)

df_merge['16zhuanghualv']=df_merge['len_order_time']/df_merge['len_request_time']
df_merge.head()

df_merge[['sku_ID','16zhuanghualv']].to_csv('16.csv')

#17 22点到2点的点击数

click_night=click[click['request_time'].dt.hour.isin([22,23,24,1,2])]
df_nightclick=pd.pivot_table(click_night,index=['sku_ID'],values=['request_time'],aggfunc=[len])

df_nightclick.columns =[s1 +'_'+ str(s2) for (s1,s2) in df_nightclick.columns.tolist()]

df_nightclick=df_nightclick.reset_index()
df_nightclick.columns=['sku_ID','17nightcick']
df_nightclick.to_csv('17.csv')

#18 渠道占比
df_qudao=pd.pivot_table(click,index=['sku_ID'],values=['request_time'],columns=['channel'],aggfunc=[len],fill_value=0)

df_qudao.columns =[s1 +'_'+str(s2)+'_'+str(s3) for (s1,s2,s3) in df_qudao.columns.tolist()]
df_qudao.reset_index(inplace=True)

df_qudao['sum']=df_qudao.iloc[:,1:6].sum(axis=1)
df_qudao['app']=df_qudao['len_request_time_app']/df_qudao['sum']
df_qudao['mobile']=df_qudao['len_request_time_mobile']/df_qudao['sum']
df_qudao['others']=df_qudao['len_request_time_others']/df_qudao['sum']
df_qudao['pc']=df_qudao['len_request_time_pc']/df_qudao['sum']
df_qudao['wechat']=df_qudao['len_request_time_wechat']/df_qudao['sum']
df_qudao.to_csv('18.csv')

#19&20 被点击次数和点击率
df_click_number=pd.pivot_table(click,index=['sku_ID'],values=['request_time'],aggfunc=[len])
df_click_number.columns =[s1 +'_'+ str(s2) for (s1,s2) in df_click_number.columns.tolist()]
df_click_number.reset_index(inplace=True)

df_click_number.columns=['sku_ID','click_number']
df_click_number['click_rate']=df_click_number['click_number']/sum(df_click_number['click_number'])
df_click_number.columns=['sku_ID','19click_number','20click_rate']
df_click_number.to_csv('19&20.csv')

#21,22 超时占比与配送时间均值
delivery_order=pd.merge(delivery,order,how='inner',on='order_ID')
delivery_order['ship_out_time']=delivery_order['ship_out_time'].astype('datetime64')
delivery_order['arr_time']=delivery_order['arr_time'].astype('datetime64')
delivery_order['time']=delivery_order['arr_time']-delivery_order['ship_out_time']

def dateDiffInHours(td):
    td=td.days* 24 + td.seconds/3600
    return td
b=[]
for a in delivery_order['time']:
    a=dateDiffInHours(a)
    b.append(a)
delivery_order['time']=b

delivery_order['promise']=pd.to_numeric(delivery_order['promise'],errors='coerce')
delivery_order['promise']=delivery_order['promise']*24
delivery_order['overdate']=np.where(delivery_order['time']>delivery_order['promise'],1,0)

df_chaoshi=pd.pivot_table(delivery_order,index=['sku_ID'],values=['package_ID'],columns=['overdate'],aggfunc=[len],fill_value=0)
df_chaoshi.columns =[s1 +'_'+str(s2)+'_'+str(s3) for (s1,s2,s3) in df_chaoshi.columns.tolist()]
df_chaoshi['21chaoshi_rate']=df_chaoshi['len_package_ID_1']/(df_chaoshi['len_package_ID_0']+df_chaoshi['len_package_ID_1'])
df_chaoshi.reset_index()
df_chaoshi['21chaoshi_rate'].to_csv('21.csv')

df_timemean=pd.pivot_table(delivery_order,index=['sku_ID'],values=['time'],aggfunc=[np.mean],fill_value=0)
df_timemean.columns =[s1 +'_'+ str(s2) for (s1,s2) in df_timemean.columns.tolist()]
df_timemean.reset_index()
df_timemean.columns=['22meantime']
df_timemean.to_csv('22.csv')

order=pd.DataFrame(pd.read_csv('order.csv',encoding='utf-8-sig',))
order['order_time']=order['order_time'].astype('datetime64')
con1=order['order_time']>=open_day
con2=order['order_time']<close_day
order=order[con1&con2]
order.head()

#最终合并数据
import pandas as pd
import numpy as np

final=pd.DataFrame(pd.read_csv('2.csv',encoding='gbk'))
final=final.rename(columns={'sku_id':'sku_ID'})
for csv in ['4.csv','1.csv','3.csv','16.csv','17.csv','18.csv','19&20.csv','21.csv','22.csv','n_std.csv']:
    df=pd.DataFrame(pd.read_csv(csv,encoding='utf-8-sig',index_col=0))
    final=pd.merge(final,df,how='inner',on='sku_ID')
final.to_csv('final.csv',encoding='gbk',index=False)

#计算销量与标准差
import pandas as pd
import numpy as np
order=pd.DataFrame(pd.read_csv('order.csv',encoding='utf-8-sig',))

df_order_number=pd.pivot_table(order,index=['sku_ID'],values=['quantity'],columns=['order_date'],aggfunc=[sum],fill_value=0)
df_order_number.head()

df_order_number.columns =[str(s3) for (s1,s2,s3) in df_order_number.columns.tolist()]

df_order_number.head()

df_order_number['2018-03-01'].head()

#number指销量，std指标准差
df_order_number['1_20number']=df_order_number.iloc[:,0:20].sum(axis=1)
df_order_number['11_30number']=df_order_number.iloc[:,10:30].sum(axis=1)
df_order_number['21_30number']=df_order_number.iloc[:,20:30].sum(axis=1)
df_order_number['1_20std']=df_order_number.iloc[:,0:20].std(axis=1)
df_order_number['11_30std']=df_order_number.iloc[:,10:30].std(axis=1)
df_order_number['21_30std']=df_order_number.iloc[:,20:30].std(axis=1)

df_order_number.iloc[:,31:37].to_csv('n_std.csv',encoding='gbk',index=True)

for csv in ['1_15.csv','final.csv','n_std.csv']:
    df=pd.DataFrame(pd.read_csv(csv,encoding='gbk',index_col=0))
    final=pd.merge(final,df,how='inner',on='sku_ID')
final.to_csv('final_1_20.csv',encoding='gbk',index=False)

import math
import pandas as pd
import numpy as np
import random 
from scipy.optimize import minimize
from scipy.stats import invgamma

# 定义fun函数，为我们需要求的目标函数
def fun(args):
    a=args
    fun=lambda x:-((x[0]*x[1])/(x[0]+x[1]))*(math.sqrt((a-x[1])/(1+x[1]))+math.sqrt((1.00001-x[0])/(a+x[0])))
    return fun

# 定义con函数，为求解的约束条件
def con(args):
    x1min, x1max, x2min, x2max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},{'type': 'ineq', 'fun': lambda x: -x[0] + x1max},{'type': 'ineq', 'fun': lambda x: x[1] - x2min},{'type': 'ineq', 'fun': lambda x: -x[1] + x2max})
    return cons
#定义final函数，得到最优解
def final(u,std,p,c):
    args = ((p+c)/c)
    args1=(0,1,0,args)
    x0=np.asarray((0.2,0.5))
    res = minimize(fun(args), x0,
                   constraints=con(args1)
                  )
    '''
    print(res.fun)
    print(res.success)

    print(res.x)
    '''
    x=res.x[0]
    y=res.x[1]
    sita=(1/(x+y))*(y*math.sqrt((args-y)/(1+y))-x*math.sqrt((1-x)/(args+x)))
    q=u+sita*std
    return q

#假设标准差属于反gamma分布，计算修正标准差后的q的上界和下界
def confidence(u,std,p,c,rmse):
    args = ((p+c)/c)
    args1=(0,1,0,args)
    x0=np.asarray((0.2,0.5))
    res = minimize(fun(args), x0,
                   constraints=con(args1)
                  )
    '''
    print(res.fun)
    print(res.success)

    print(res.x)
    '''
    x=res.x[0]
    y=res.x[1]
    sita=(1/(x+y))*(y*math.sqrt((args-y)/(1+y))-x*math.sqrt((1-x)/(args+x)))
    

    a=(std**2/rmse**2)+2
    b=(std**3/rmse**2)+std
    std1=invgamma.ppf(0.05,a,scale=b)#下界
    std2=invgamma.ppf(0.95,a,scale=b)#上界
    
    q1=u+sita*std1
    q2=u+sita*std2
    return q1,q2

#测试函数
if __name__=='__main__':
    print(final(8,math.sqrt(267.026816),0.034518,0.014312))
    print(confidence(8,math.sqrt(267.026816),0.034518,0.014312,100.672400))

#带入我们求出得到u，std，cu，co进行计算
pj=pd.DataFrame(pd.read_csv('Pj.csv',encoding='utf-8-sig',))
ci=pd.DataFrame(pd.read_csv('Ci.csv',encoding='utf-8-sig',))
u0=pd.DataFrame(pd.read_csv('u.csv',encoding='utf-8-sig',))
var0=pd.DataFrame(pd.read_csv('var.csv',encoding='utf-8-sig',))

u0=pd.merge(u0,var0,how='inner')
u0=pd.merge(u0,pj,how='inner',on='sku_ID')
u0=pd.merge(u0,ci,how='inner',on='dc')

u0['q']=0
u0['a']=(u0['Pj']+u0['C'])/u0['C']

#对dataframe进行循环计算结果
for i in range(len(u0['dc'])):
    try:
        u0.iloc[i,10]=final(u0.iloc[i,3],math.sqrt(u0.iloc[i,4]),u0.iloc[i,6],u0.iloc[i,9])
    except:
        u0.iloc[i,10]=' '
    else:
        pass

#u0.to_csv('plus.csv')

#按照每个DC上的方差对rmse进行分配
rmse=pd.DataFrame(pd.read_csv('n_std.csv',encoding='utf-8-sig',))
rmse['var']=rmse['1_20std']**2
#df_pivot=pd.pivot_table(rmse,index=['sku_ID'],values=['var'],aggfunc=[sum],fill_value=0)

rmse['var_rmse']=25*rmse['var']/(sum(rmse['var'])/len(rmse['var']))
rmse

std_20=pd.DataFrame(pd.read_csv('var_quan.csv',encoding='utf-8-sig',))
rmse_varmse=rmse[['sku_ID','var_rmse']]
rmse_varmse=pd.merge(std_20,rmse_varmse,how='inner',on='sku_ID')

rmse_varmse=rmse_varmse.rename(columns={'dc_des':'dc'})
rmse_varmse['rmse']=rmse_varmse['var']*rmse_varmse['var_rmse']

#rmse_varmse.to_csv('rmse.csv')

# 求置信区间
u0=pd.merge(u0,rmse_varmse[['sku_ID','dc','rmse']],how='inner')
u0['q1']=0
u0['q2']=0

for i in range(len(u0['dc'])):
    try:
        if u0.iloc[i,12]==0 or u0.iloc[i,4]==0:
            u0.iloc[i,13]=u0.iloc[i,10]
            u0.iloc[i,14]=u0.iloc[i,10]
        else:
            u0.iloc[i,13]=confidence(u0.iloc[i,3],math.sqrt(u0.iloc[i,4]),u0.iloc[i,6],u0.iloc[i,9],u0.iloc[i,12])[0]
            u0.iloc[i,14]=confidence(u0.iloc[i,3],math.sqrt(u0.iloc[i,4]),u0.iloc[i,6],u0.iloc[i,9],u0.iloc[i,12])[1]
    except:
        u0.iloc[i,13]=' '
        u0.iloc[i,14]=' '
    else:
        pass

u0.to_csv('confidence.csv')

# 求scarf论文中的上界最优解
scarf_q=pd.DataFrame(pd.read_csv('confidence.csv',encoding='utf-8-sig',))
scarf_q

# 用论文的q公式进行计算
for i in range(len(scarf_q['miu'])):
    scarf_q.loc[i,'q*3']=scarf_q.loc[i,'miu']+(math.sqrt(scarf_q.loc[i,'var'])/2)*((math.sqrt((scarf_q.loc[i,'Pj']+scarf_q.loc[i,'C'])/scarf_q.loc[i,'C']))-math.sqrt(scarf_q.loc[i,'C']/(scarf_q.loc[i,'Pj']+scarf_q.loc[i,'C'])))

#scarf_q.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)#删去出现的无用列

#导出最终的各种情况的最优销量Q
huibiao=scarf_q[['sku_ID','dc','q1','q2','q','q*2','q*3']]
huibiao.to_csv('huibiao.csv')

# 抽取20个样本进行绘图
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.plot(range(20),scarf_q['q1'][:20],label='q下界')
plt.plot(range(20),scarf_q['q2'][:20],label='q上界')
plt.plot(range(20),scarf_q['q*3'][:20],label='scarf_q*')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.fill_between(range(20),scarf_q['q1'][:20],scarf_q['q2'][:20],color='lightyellow')
plt.legend()
plt.title('q取样比较图')
plt.xlabel("序号", fontsize=14)
plt.ylabel("分配量q", fontsize=14)
#plt.savefig('QN.png')
plt.show()

