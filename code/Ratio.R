Data=read.csv('final1-20.csv')
data0=Data[1:20,]
ke_skuid=data.frame(data0$sku_ID)#保留销量最高的前20个SKU
colnames(ke_skuid)=c('sku_ID')
ke_skuid$sku_ID=as.character(ke_skuid$sku_ID)
order_data=read.csv('JD_order_data.csv')
order_data$sku_ID=as.character(order_data$sku_ID)
dcdata=merge(ke_skuid,order_data,by='sku_ID',all=F)#保留order表中相应20个SKU数据
sum(dcdata$gift_item)#没有赠品
Demand=data.frame(dcdata$sku_ID,dcdata$quantity,dcdata$dc_des)
colnames(Demand)=c('sku_ID','quantity','dc_des')
sum(is.na(Demand$dc_des))#无缺失值
dis_dc=aggregate(Demand$quantity,by=list(Demand$dc_des,Demand$sku_ID),FUN=sum)#计算20天内每一SKU在每一dc上的销量
colnames(dis_dc)=c('dc_des','sku_ID','quantity')
predictdata=read.csv('predict21-30(1).csv')#导入预测数据
predictdata=predictdata[,-c(1)]
colnames(predictdata)=c('sku_ID','u','var')
dis_dc=merge(dis_dc,predictdata,by='sku_ID')#合并表格
distribution=aggregate(dis_dc$quantity,by=list(dis_dc$sku_ID),FUN=sum)#计算20天内每一SKU的销量
colnames(distribution)=c('sku_ID','sum')
dis_dc=merge(dis_dc,distribution,by='sku_ID')
ratio=dis_dc$quantity/dis_dc$sum#销量分配权重
dis_dc=data.frame(dis_dc,ratiodata)
miu=dis_dc$u*dis_dc$ratio#未来10天miuij分配
miudata=data.frame(dis_dc$sku_ID,dis_dc$dc_des,miu)
colnames(miudata)=c('sku_ID','dis_dc','miu')
write.csv(miudata,file = 'miu.csv')
#算方差sigma
sigdata=aggregate(Demand$quantity,by=list(Demand$dc_des,Demand$sku_ID),FUN=sd)#每一SKU在每一dc上的销量标准差
sigdata[is.na(sigdata$std,3)]=0
colnames(sigdata)=c('dc_des','sku_ID','std')
write.csv(sigdata,file='std0.csv')#得到sigmaij
sigdata=aggregate(Demand$quantity,by=list(Demand$dc_des,Demand$sku_ID),FUN=var)#每一SKU在每一dc上的销量方差
data1=dis_dc[,c(1:3)]
dis_dc=data.frame(data1,sigdata$var)
colnames(dis_dc)=c('sku_ID','dc_des','quantity','var')
dis_dc[is.na(dis_dc$var),4]=0
varsum=aggregate(dis_dc$var,by=list(dis_dc$sku_ID),FUN=sum)#每一SKU的销量方差
colnames(varsum)=c('sku_ID','var')
dis_dc=merge(dis_dc,varsum,by='sku_ID')
colnames(dis_dc)=c('sku_ID','dc_des','quantity','var','varsum')
sigratio=dis_dc$var/dis_dc$varsum#得到方差分配权重
dis_dc1=data.frame(dis_dc1,sigratio)
var=dis_dc1$sigratio*dis_dc1$var.y#未来10天每一SKU在每一dc上的销量方差
var0=data.frame(dis_dc$sku_ID,dis_dc$dc_des,var)
colnames(var0)=c('sku_ID','dis_dc','var')
write.csv(var0,file='var0.csv')