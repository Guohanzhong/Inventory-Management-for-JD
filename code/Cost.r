#计算21-30号sku在每个dc的总销量
setwd("E:\\Rcode_Lecture\\JD_com\\Raw_Data")
order<-read.csv("JD_order_data.csv",header=T)
order<-order[(order$sku_ID %in% Pj$sku_ID),]
order$order_date<-as.Date(order$order_date)
order<-order[order$order_date>=as.Date("2018-03-21"),]
order<-order[order$order_date<=as.Date("2018-03-30"),]
order<-order[order$gift_item==0,]
Dij<-aggregate(order$quantity,by=list(order$sku_ID,order$dc_des),FUN=sum)
sku_ID<-data.frame(Pj$sku_ID)
colnames(sku_ID)<-c('sku_ID')
sku_ID$sku_ID<-as.character(sku_ID$sku_ID)
order$sku_ID<-as.character(order$sku_ID)
dc<-merge(sku_ID,order,by='sku_ID',all=F)#保留前20个sku数据
sum(dc$gift_item)#没有赠品
Demand<-data.frame(dc$sku_ID,dc$quantity,dc$dc_des)
colnames(Demand)=c('sku_ID','quantity','dc_des')
sum(is.na(Demand$dc_des))#无缺失值
dis_dc=aggregate(Demand$quantity,by=list(Demand$dc_des,Demand$sku_ID),FUN=sum)
colnames(dis_dc)=c('dc_des','sku_ID','quantity')
write.csv(dis_dc,file = "Dij.csv",row.names = F)

#已有20个sku
#20个商品j的价格，第i的DC离其他DC的平均距离
setwd("E:\\Rcode_Lecture\\JD_com")
Ci<-read.csv("Ci.csv",header=T)
Ci<-Ci[,-c(2,3)]#第i个DC离其他dc的平均距离
Pj<-read.csv("Pj.csv",header=T)
Pj<-Pj[,-2]#商品的sku和价格
Dj<-read.csv("FinalTable1-20.csv",header=T)
Dj<-Dj[,c(1,34)]
Dj<-Dj[(Dj$sku_ID %in% Pj$sku_ID),]#每个sku的总销量
Xij<-read.csv("Huibiao.csv",header = T)
Dij<-read.csv("Dij.csv",header=T)
Xij<-Xij[(Xij$sku_ID %in% Pj$sku_ID),]#每个sku的总销量
colnames(Dij)<-c("dc","sku_ID","quantity")
Xj<-data.frame(Pj$sku_ID)
Xj[,c(2,3,4)]<-0
colnames(Xj)<-c("sku_ID","X1","X2","X3")
Xj$X1<-aggregate(Xij$q,by=list(Xij$sku_ID),FUN=sum)
Xj$X2<-aggregate(Xij$q.2,by=list(Xij$sku_ID),FUN=sum)
Xj$X3<-aggregate(Xij$q.3,by=list(Xij$sku_ID),FUN=sum)

#计算利润损失
prof_lost<-merge(Pj,Dj,by="sku_ID")
prof_lost<-merge(Xj,prof_lost,by="sku_ID")
prof_lost[,c(7,8,9)]<-0
cnames<-c("sku_ID","X1","X2","X3","Pj","Dj","sum1","sum2","sum3")
colnames(prof_lost)<-cnames

prof_lost$sum1<-(prof_lost$Dj-prof_lost$X1)*prof_lost$Pj
prof_lost$sum2<-(prof_lost$Dj-prof_lost$X2)*prof_lost$Pj
prof_lost$sum3<-(prof_lost$Dj-prof_lost$X3)*prof_lost$Pj
for(i in 1:nrow(prof_lost))
{
  if(prof_lost[i,7]<0)prof_lost[i,7]<-0
  if(prof_lost[i,8]<0)prof_lost[i,8]<-0
  if(prof_lost[i,9]<0)prof_lost[i,9]<-0

}
sum_prof_lost_1<-sum(prof_lost$sum1)
sum_prof_lost_2<-sum(prof_lost$sum2)
sum_prof_lost_3<-sum(prof_lost$sum3)

#计算运输成本
#假设第一列是dc编号，第二列是sku，第三列是dij，第四列是xij
trans_cost<-merge(Dij,Xij)
trans_cost<-merge(trans_cost,Ci)
trans_cost[,c(8,9,10)]<-0
cnames<-c("dc","sku_ID","quantity","x1","x2","x3","C","sum1","sum2","sum3")
colnames(trans_cost)<-cnames
trans_cost$sum1<-trans_cost$C*(trans_cost$x1-trans_cost$quantity)
trans_cost$sum2<-trans_cost$C*(trans_cost$x2-trans_cost$quantity)
trans_cost$sum3<-trans_cost$C*(trans_cost$x3-trans_cost$quantity)
for(i in 1:nrow(trans_cost))
{
  if(trans_cost[i,8]<0)trans_cost[i,8]<-0
  if(trans_cost[i,9]<0)trans_cost[i,9]<-0
  if(trans_cost[i,10]<0)trans_cost[i,10]<-0
}
sum_cost_1<-sum(trans_cost$sum1)
sum_cost_2<-sum(trans_cost$sum2)
sum_cost_3<-sum(trans_cost$sum3)

cost1<-sum_prof_lost_1+sum_cost_1
cost2<-sum_prof_lost_2+sum_cost_2
cost3<-sum_prof_lost_3+sum_cost_3

write.csv(prof_lost,"prof_lost.csv")
write.csv(trans_cost,"trans_cost.csv")