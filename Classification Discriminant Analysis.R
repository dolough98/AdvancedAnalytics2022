setwd("C:\\Users\\35383\\Documents\\Advanced Analytics in Business")
BankCust<-read.csv("train3wtfc")

colSums(is.na(BankCust))
set.seed(197)
trainsize = 50000
train = sample(nrow(BankCust), trainsize)
dim(BankCust)
BankCust1<- as.data.frame(BankCust)
head(BankCust1)


train <- subset(BankCust,select = -c(X))
dim(train)
head(train)
typeof(train)

trains = sample(nrow(train), trainsize)
length(trains)
dim(trains)
data.train<-BankCust[trains,1:56]
data.test<-BankCust[-trains,1:56]
target.train<-BankCust[trains,57]
target.test<-BankCust[-trains,57]

# Centering and standardizing the data
cdata.train<-scale(data.train,center=TRUE,scale=FALSE)
cdata.test<-scale(data.test,center=TRUE,scale=FALSE)
zdata.train<-scale(data.train,center=TRUE,scale=TRUE)
zdata.test<-scale(data.test,center=TRUE,scale=TRUE)

lda.out<-lda(cdata.train,target.train)
print(lda.out)

boxM(BankCust2[,1:56], BankCust[,"target"])
#equal covariance matrices across groups is not supported


#predictions training data
pred.train<-predict(lda.out,cdata.train, LOO = TRUE)
pred.train$posterior[1:5,]
#Bayes classifier
tab<-table(target.train,pred.train$class)
lda.train.equal<-performance(tab)
lda.train.equal
tab

## Using Bayes Classifier only 22/2500 = 0.008 of the customers are classified as buyers.137/146 are buyers who are incorrectly classified. 13/2354 are non buyers who are incorrectly classified.

#unequal classification costs
class.train<-ifelse(pred.train$posterior[,2]*20>=pred.train$posterior[,1]*1,1,0)
tab<-table(target.train,class.train)
lda.train.unequal<-performance(tab)
lda.train.unequal
print(tab)
## After accounting for classification costs 782/2500 = 31% of the observations are classified as buyers. This is higher than with the bayes classifier as classifying a non-buyer as a buyer costs less than classifying a buyer as a non-buyer.
# classification error

#predictions test data
pred.test<-predict(lda.out,cdata.test, LOO=TRUE)
#Bayes classifier
tab<-table(target.test,pred.test$class)
tab

lda.test.equal<-performance(tab)
lda.test.equal
#unequal classification costs
class.test<-ifelse(pred.test$posterior[,2]*20>=pred.test$posterior[,1]*1,1,0)
tab<-table(target.test,class.test)
tab
lda.test.unequal<-performance(tab)
lda.test.unequal

pred.test$posterior[,"1"]

roc_lda <- roc(target.test, pred.test$posterior[,"1"])
plot(roc_lda,col="red", lwd=3, main="ROC curve LDA")
auc(roc_lda)


qda.train.equal<-qda(cdata.train,target.train)
qda.train.unequal<-(NA)
qda.test.equal<-(NA)
qda.test.unequal<-(NA)
#qda
QDA.out<-qda(cdata.train[,1:37],target.train)
## The error indicated here that there is rank deficiency. that some variables are collinear and one of more covariances matrices cannot be inverted to obtain the estimates in group yes

library(klaR)


set.seed(1)
hdda.out <- hdda(cdata.train,target.train,model="all",d_select="BIC")

pred.train<-predict(hdda.out,cdata.train)
tab<-table(target.train,pred.train$class)
tab
hdda.train.equal<-performance(tab)
hdda.train.equal

#unequal classification costs
class.train<-ifelse(pred.train$posterior[,2]*20>=pred.train$posterior[,1]*1,1,0)
tab<-table(target.train,class.train)
tab
hdda.train.unequal<-performance(tab)
hdda.train.unequal

#use LOO=TRUE option with the model selected using the training data to compute LOOCV classifications
pred.test<- predict(hdda.out,cdata.test, LOO=TRUE)
tab<-table(target.test,pred.test$class)
tab
hdda.test.equal<-performance(tab)
hdda.test.equal
# Unequal classification costs
class.test<-ifelse(pred.test$posterior[,2]*20>=pred.test$posterior[,1]*1,1,0)
tab<-table(target.test,class.test)
tab
hdda.test.unequal<-performance(tab)
hdda.test.unequal

roc_hdda <- roc(target.test, pred.test$posterior[,"1"])
plot(roc_hdda,col="red", lwd=3, main="ROC curve QDA")
auc(roc_hdda)



## Analysis of standardized raw data with PCA
PCA.out <- prcomp(zdata.train)
# Eigenvalues
round(PCA.out$sdev^2,2)
# Proportion of explained variance per component
round(PCA.out$sdev^2/83,3)
# Compute Component loadings
A <- PCA.out$rotation%*%diag(PCA.out$sdev)
A[,1]
# plot of loadings for first two components
# use package for pointlabels
library(maptools)
plot(A[,1:2],xlim=c(-1,1),ylim=c(-1,1),xlab="PC1",ylab="PC2")

#compute unstandardized component scores
Zun<-zdata.train%*%PCA.out$rotation[,1:55]
# compute stadardized component scores
Zs<-zdata.train%*%PCA.out$rotation[,1:55]%*%diag(1/PCA.out$sdev[1:82])
# plot component scores for first 2 unstandardized components
plot(Zun[,1:2],xlim=c(-10,10),ylim=c(-10,10),xlab="PC1",ylab="PC2")

## How Many components
screeplot(PCA.out,type = "lines",ylim = c(0,10), npcs = min(83, length(PCA.out$sdev)))
screeplot(PCA.out,type = "lines",ylim = c(1,3), npcs = min(39, length(PCA.out$sdev)))
# Using Kaisers rule we only use the first 30 components

# Horn's procedure to choose the number of components
# using centile=0 we compute the mean of the bootstrapped eigenvalues
# Estimated bias= mean of bootstrapped eigenvalue minus 1
# Adjusted eigenvalue= unadjusted eigenvalue- bias
set.seed(589)
library(paran)
paran(zdata.train,iterations=1000,graph=TRUE,cfa=FALSE,centile=0)
# Using horn's procedure we retain components for which the observed eigenvalue > mean of random eigenvalue. This is equivalent to retaining components for adjusted eigenvalue is > 1.
# Applying horn's procedure we retain the first 31 components

# using centile=95 we compute the 95% percentile of the
# distribution of the bootstrapped eigenvalues
# Estimated bias=p95 of distribution bootstrapped eigenvalue minus 1
# Adjusted eigenvalue= unadjusted eigenvalue - bias
set.seed(897)
paran(zdata.train,iterations=1000,graph=TRUE,cfa=FALSE,centile=95) 
# USing a conservative procedure we still retain the first 31 components


# compute the proportion of
# variance accounted for (VAF) in each
# variable by the first 31 principal
# components
sort(round(diag(A[,1:24]%*%t(A[,1:24])),2))
# The three component that explains less than 50% of the variance is MGODPR(Roman Catholic), MINK123 (INcome>123000) and MAANTHUI(Number of houses 1-10).

loading<-round(PCA.out$rotation%*%diag(PCA.out$sdev),2)
loading[,1]

pred<-predict(PCA.out)
round(pred[,1],2)
plot(pred)
#PCA.out
#biplot(PCA.out,pc.biplot = TRUE)


PCA_Cvn <- prcomp(scale(BankCust[,1:56],center=TRUE,scale=TRUE))
library(factoextra)
dim(PCA_Cvn$x)

typeof(PCA_Cvn$x)
train_Cvn <- data.frame(PCA_Cvn$x[trains,1:56])
#train_Cvn
test_Cvn <- data.frame(PCA_Cvn$x[-trains,1:56])
test_data <- data.frame(test_Cvn)

screeplot(PCA_Cvn, type = "lines", npcs = 83)
screeplot(PCA_Cvn, type = "lines", npcs = 31)


y_t <- (BankCust[,57])
y_train <- y_t[trains]
y_test <- y_t[-trains]
#head(y_train)
#head(y_test)
#head(train_Cvn)

train_data = data.frame(label = y_train, train_Cvn)


###########
#bagging
###########
set.seed(1)
ctrain<-cbind(cdata.train,target.train)
bag.cvn=randomForest(as.factor(target.train)~.,data=ctrain,mtry=24,ntree=1000,importance=TRUE)
bag.cvn

#plot oob error
par(cex=1.2)
plot(1:100,bag.cvn$err.rate[1:100,1],xlab="Number of iterations",ylab="OOB error",pch='',main="OOB error")
lines(1:500,bag.cvn$err.rate[1:500,1],col="red")
#plot variable importance
importance(bag.cvn,plot=TRUE)
varImpPlot(bag.cvn,type=2,cex=1.2)
length(bag.cvn)
#predictions training data 
pred.train<-predict(bag.cvn,newdata=ctrain,type = "prob")
err2(ctrain[,84],bag.cvn$predicted) 
length(pred.train)
#Bayes classifier
class.train<-ifelse(pred.train[,2]>0.5,1,0)
tab<-table(target.train,class.train)
tab
bag.train.equal<-performance(tab)
bag.train.equal
#unequal classification costs

class.train<-ifelse(pred.train[,2]*20>=pred.train[,1]*1,1,0)
length(class.train)
tab1<-table(target.train,class.train)
tab1
bag.train.unequal<-performance(tab1)
bag.train.unequal

#predictions test data
ctest <- cbind(cdata.test,target.test)
length(pred.test)
pred.test<-predict(bag.cvn,newdata=ctest)
err2(ctest[,57],pred.test)

#Bayes classifier
pred.test2<-predict(bag.cvn,newdata=ctest,type = "prob")
class.test<-ifelse(pred.test2[,2]>0.5,1,0)
tab<-table(target.test,class.test)
tab
bag.test.equal<-performance(tab)
bag.test.equal
#unequal classification costs
class.test<-ifelse(pred.test2[,2]*20>=pred.test2[,1]*1,1,0)
tab2<-table(target.test,class.test)
tab2
bag.test.unequal<-performance(tab2)
bag.test.unequal

dim(pred.test2[,1])
length(pred.test[1:13697])
length(target.test)


roc_rf <- roc(target.test, pred.test2[1:13697])
plot(roc_rf,col="red", lwd=3, main="ROC curve QDA")
auc(roc_rf)
