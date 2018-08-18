# SET WORKING DIRECTORY AND FILE PATH

# MY_WD - the working directory
# PATH - the path that the file is stored

# INSTALL ALL DEPENDENCIES
#install.packages(rpart.plot)
#install.packages(e1071)
#install.packages(ROCR)

# RUN THE CODE

setwd(MY_WD)
myData - read.csv(file=PATH)
head(myData)

# class frequencies (distribution)
table(myData$left)


# changing categorical data to numerical data
myData$sales- factor(myData$sales,levels = c(sales,accounting,hr,technical,support,management,IT,product_mng,marketing,RandD),labels = c(1,2,3,4,5,6,7,8,9,10))
myData$salary - factor(myData$salary,levels=c(low,medium,high),labels =c(0,1,2))


library(rpart.plot)
# splitting to train and test set
set.seed(1)
train.idx - sample(1nrow(myData), size = round(0.8  nrow(myData)), replace = FALSE)
train.set - myData[train.idx,]
test.set - myData[-train.idx,]

# creating a decision tree
dt - rpart(left ~ ., data =train.set , method = 'class')

# plot tree in 2 different ways
plot(dt)
text(dt, pretty = 0)
rpart.plot(dt)


library('ROCR')

#ROC curve
dtPred - predict(dt,newdata=test.set,type=prob)[,2]
table(dtPred,test.set$left)
pred2 = prediction(dtPred,test.set$left)
plot(performance(pred2,tpr,fpr))
abline(0, 1, lty = 2)

# AUC
auc - unlist(attr(performance(pred2, auc), y.values))
print(auc)
legend(bottomright, sprintf(%.3f,auc), title = AUC)