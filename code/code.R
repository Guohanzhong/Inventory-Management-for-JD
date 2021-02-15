setwd("E:\\大三上\\数据挖掘与机器学习\\小组作业\\jd\\Jd_com_data")
order=read.csv("JD_order_data.csv")
user=read.csv("JD_user_data.csv")
delivery=read.csv("JD_delivery_data.csv")
fi=read.csv("final1-20（排序）.csv")

### 选择非礼物的数据，去掉gift item这列
order=order[order$gift_item==0,]
order=order[,-15]


###拆分为前2个10天和后2个10天
order$order_date=as.Date(order$order_date)
orderx=order[order$order_date!=as.Date("2018-03-31"),]
order=orderx
#order$order_date=as.character(order$order_date)
order1=order[order$order_date<=as.Date("2018-03-20"),] #前2个10天的订单数据
order2=order[order$order_date>=as.Date("2018-03-11"),] #后2个10天的订单数据

product=order$sku_ID[!duplicated(order$sku_ID)] #所有的sku_id

x=merge(t1,order3,by="sku_ID")




                  
a1=data.frame(product) #a1储存前2个10天被出售的商品属性
a2=data.frame(product) #a2储存后2个10天被出售的商品属性
# 先a1 前2个10天
a1[,2:5]=0
a2[,2:5]=0
cname=c("product","前2个10天销量","前2个10天销售额","前2个10天购买者plus会员比例","前2个10天购买者平均购买力")
colnames(a1)=cname
colnames(a2)=cname

for (i in c(1:nrow(t1))){
  t1[i,33]=sum(x[x$sku_ID==t1[i,]$sku_ID,]$quantity)
}

colnames(t1)[33]="后10天销售量"
write.csv(t1,file = "train.csv")

for (i in c(1:nrow(a1))){
  #b是包含该商品的所有订单
  b=order1[order1$sku_ID==a1[i,1],]
  a1[i,2]=sum(b$quantity)
  a1[i,3]=sum(b$final_unit_price)
  #d是买了该商品的所有用户
  d=user[(user$user_ID %in% b$user_ID),]
  a1[i,4]=(sum(d$plus))/nrow(d)
  a1[i,5]=mean(d$purchase_power)
}

# 搞a2 后2个10天
for (i in c(1:nrow(a2))){
  #b是包含该商品的所有订单
  b=order2[order2$sku_ID==a2[i,1],]
  a2[i,2]=sum(b$quantity)
  a2[i,3]=sum(b$final_unit_price)
  #d是买了该商品的所有用户
  d=user[(user$user_ID %in% b$user_ID),]
  a2[i,4]=(sum(d$plus))/nrow(d)
  a2[i,5]=mean(d$purchase_power)
}


#######计算不同dc之间的平均配送时间，用前20天的order数据######
ordel=merge(order1,delivery,by="order_ID")

table(ordel$dc_ori)
table(ordel$dc_des)


ctable=as.data.frame(c(1:67))
ctable[,c(2:4)]=0

cname2=c("dc","meanTime(sec)","meanTime(h)","C")
colnames(ctable)=cname2

library(lubridate)
ordel$ship_out_time=ymd_hms(ordel$ship_out_time)
ordel$arr_station_time=ymd_hms(ordel$arr_station_time)
for (i in c(1:67)){
    o=ordel[(ordel$dc_ori==i)|(ordel$dc_des==i),]
    ctable[i,2]=mean(as.duration(o$arr_station_time-o$ship_out_time))
}
ctable[,3]=ctable[,2]/3600
ctable[,4]=ctable[,3]/sum(ctable[,3])

write.csv(ctable,"cost.csv",row.names = F)

######计算不同商品的P########
ptable=data.frame(fi[1:20,1])
ptable[,2:3]=0
cnames=c("sku_ID","P","Pj")
colnames(ptable)=cnames
for(i in c(1:20)){
  b=order1[as.character(order1$sku_ID)==as.character(ptable[i,1]),]
  ptable[i,2]=mean(b$final_unit_price)
}
for (i in c(1:20)){
  ptable[i,3]=ptable[i,2]/sum(ptable$P)
}
write.csv(ptable,file = "Pcost.csv",row.names = F)

#######算xij 启发式算法#######
u=read.csv("miu(1).csv")
s=read.csv("var1(1).csv")
P=read.csv("Pj.csv")
C=read.csv("Ci.csv")

k=merge(s,u,by=c("sku_ID","dc"))
k=k[,-4]
k[,5:7]=0
colnames(k)[5:7]=c("x1","x2","x3")
p=merge(k,P,by="sku_ID")
p=merge(k,C,by="dc")

#算xij
k[,6]=(1+1/67)*k[,5]
k[,7]=k[,5]+(k[,4])^0.5/67
k[,8]=(1+k[,3]/(67*k[,9]))*k[,5]

# 把xij提取出来
best=k[,c(1,2,6,7,8)]
write.csv(best,file="xij.csv",row.names = F)
