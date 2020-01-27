#(i) LOGISTIC REGRESSION

#read wine dataset
mywine <- read.table(file.choose(), header=FALSE)
dim(mywine)
summary(mywine)

#rename columns 
colnames(mywine) <- c("Facidity", "Vacidity", "Citric", "sugar", "chlorides", "freeSD", 
                      "totalSD", "density", "pH", "Sulphates", "Alcohol", "Y")

#Standardize data
data.numcols <- mywine[,sapply(mywine, is.numeric)]
means <- apply(data.numcols,2,mean)
standarddeviations <- apply(data.numcols,2,sd)
mywineSTAN <- as.data.frame(scale(data.numcols,center=means,scale=standarddeviations))
mywineSTAN$Y <- mywine$Y

#change Y to take value 0 or 1
str(mywineSTAN)

#Reshuffle the dataset
set.seed(123)
mymywineSTAN <- mywineSTAN[sample(nrow(mywineSTAN)),]
testwine <- mymywineSTAN[1:240,]
miniwine <- mymywineSTAN[240: nrow(mymywineSTAN),]
summary(miniwine)
summary(testwine)

#Regression model
model1 <- glm(Y ~ ., family='binomial', miniwine )
summary(model1)
logLik(model1)

#prediction logistic regression
predictlog <- predict(model1, testwine[,-12],type="response")
classes <- factor(round(predictlog),levels=c(0,1),labels=c("good", "notgood"))
classificationlog <- table(classes,testwine[,12])
print (classificationlog)

#accuracy logistic regression
sum(diag(classificationlog))/sum(classificationlog)


#(ii) SUPPORT VECTOR MACHINES

#tuning with SVM
library(e1071)
set.seed(123)
tunedmodellinear <- tune.svm(Y~.,data = miniwine, cost=10^(-3:3),kernel="polynomial")
tunedmodellinear$best.parameters[[1]]
tunedmodellinear$best.model
finalmodellinear <- svm(Y~.,data=miniwine,cost=tunedmodellinear$best.parameters[[1]], kernel="polynomial")
print(finalmodellinear)

#test accuracy / predict on testwine
predictionlinear <- predict(finalmodellinear,testwine[,-12])
classificationtable <- table(pred=predictionlinear,testwine[,12])
sum(diag(classificationtable))/sum(classificationtable)
#misclassification rate
1-sum(diag(classificationtable))/sum(classificationtable)
classificationtable

#tuning with RBF
set.seed(123)
tunedmodelRBF <- tune.svm(Y~.,data=miniwine,gamma=2^(-2:2),cost=10^(0:2))
tunedmodelRBF$best.parameters[[1]]
tunedmodelRBF$best.parameters[[2]]
tunedmodelRBF$best.parameters
finalmodelRBF <- svm(Y~.,data=miniwine,gamma=tunedmodelRBF$best.parameters[[1]],cost=tunedmodelRBF$best.parameters[[2]])
print(finalmodelRBF)

#test accuracy / predict on testwine
predictionRBF <- predict(finalmodelRBF,testwine[,-12])
classificationtable <- table(pred=predictionRBF,testwine[,12])
accuracyRBF <- sum(diag(classificationtable))/sum(classificationtable)
print(accuracyRBF)
classificationtable

#(iii) CLASSIFICATION TREE
library(tree)
library(randomForest)
table(miniwine$Y)
table(testwine$Y)

#Build tree using cross validation
mytree <- tree(Y~.,miniwine)
plot(mytree)
text(mytree)
summary(mytree)
set.seed(123)
mycrossval <- cv.tree(mytree,FUN=prune.tree,K=10)
print(mycrossval)

#Pruned tree
mybestsize <- mycrossval$size[which(mycrossval$dev==min(mycrossval$dev))] 
myprunedtree <- prune.tree(mytree,best=mybestsize)
print(myprunedtree)
plot(myprunedtree)
text(myprunedtree)
summary(myprunedtree)

#Accuracy pruned test
myprediction <-predict(myprunedtree,testwine, type = "class")
classificationtable <- table(myprediction,testwine[,12])
sum(diag(classificationtable))/sum(classificationtable)
classificationtable

#(iv) RANDOM FOREST
set.seed(123)
myrf <- randomForest(Y~.,miniwine,ntree=500,mtry=3,importance=TRUE)
importance(myrf)
varImpPlot(myrf)

#Accuracy Random Forest
myprediction <- predict(myrf, testwine, type='class')
classificationtable <- table(myprediction,testwine[,12])
accuracyRF <- sum(diag(classificationtable))/sum(classificationtable)
sum(diag(classificationtable))/sum(classificationtable)
print(classificationtable)
table(classificationtable)

#(v) K-NEAREST NEIGHBORS
library(FNN)
myxtrain <- miniwine[,-12]
myytrain <- miniwine[,12]
myxtesting <- testwine[,-12]
myytesting <- testwine[,12]
myk1nn <- knn(train= myxtrain, test= myxtesting, cl= myytrain, k=1)

#accuracy KNN
myaccuracytablek1nn <- table (myk1nn,myytesting)
mytestingaccuracyk1nn <- sum(diag(myaccuracytablek1nn))/sum(myaccuracytablek1nn)
print (mytestingaccuracyk1nn)

#leave one out
mycvack1nn <- knn.cv(train= myxtrain, cl= myytrain, k=1)
mycvactablek1nn <- table (mycvack1nn, myytrain)
mycvack1nn <- sum(diag(mycvactablek1nn))/sum(mycvactablek1nn)
bestk = 1 
bestaccuracy = mycvack1nn
mycvack3nn <- knn.cv(train= myxtrain, cl= myytrain, k=3)
mycvactablek3nn <- table (mycvack3nn, myytrain)
mycvack3nn <- sum(diag(mycvactablek3nn))/sum(mycvactablek3nn)
if(bestaccuracy< mycvack3nn) bestk=3
if(bestaccuracy< mycvack3nn) bestaccuracy = mycvack3nn
print(mycvactablek3nn)


#(vi) PREDICT USING VALIDATION DATASET

myvalidation <- read.table(file.choose(), header=FALSE)
dim(myvalidation)
summary(myvalidation)

#Standardize data
data.numcols <- myvalidation[,sapply(myvalidation, is.numeric)]
means <- apply(data.numcols,2,mean)
standarddeviations <- apply(data.numcols,2,sd)
myvalidationSTAN <- as.data.frame(scale(data.numcols,center=means,scale=standarddeviations))
myvalidationSTAN$Y <- myvalidation$V12

#rename columns 
colnames(myvalidationSTAN) <- c("Facidity", "Vacidity", "Citric", "sugar", "chlorides", "freeSD", 
                                "totalSD", "density", "pH", "Sulphates", "Alcohol", "Y")

#prediction logistic regression
predictlog <- predict(model1, mywineSTAN[,-12],type="response")
classes <- factor(round(predictlog),levels=c(0,1),labels=c("good", "notgood"))
classificationlog <- table(classes,mywineSTAN[,12])
print (classificationlog)
sum(diag(classificationlog))/sum(classificationlog)

#SVM linear
set.seed(123)
predictionlinear <- predict(finalmodellinear,myvalidationSTAN[,-12])
classificationtable <- table(pred=predictionlinear,myvalidationSTAN[,12])
sum(diag(classificationtable))/sum(classificationtable)

#RBF 
predictionRBF <- predict(finalmodelRBF,myvalidationSTAN[,-12])
classificationtable <- table(pred=predictionRBF,myvalidationSTAN[,12])
sum(diag(classificationtable))/sum(classificationtable)

#pruned tree
myprediction <-predict(myprunedtree,myvalidationSTAN, type = "class")
classificationtable <- table(myprediction,myvalidationSTAN[,12])
sum(diag(classificationtable))/sum(classificationtable)

#random forest
myprediction <- predict(myrf, myvalidationSTAN, type='class')
classificationtable <- table(myprediction,myvalidationSTAN[,12])
sum(diag(classificationtable))/sum(classificationtable)

#KNN
myxtrain <- mywineSTAN[,-12]
myytrain <- mywineSTAN[,12]
myxtesting <- myvalidationSTAN[,-12]
myytesting <- myvalidationSTAN[, 12]

myk1nn <- knn(train= myxtrain, test= myxtesting, cl= myytrain, k=100)

myaccuracytablek1nn <- table (myk1nn, myytesting)
mytestingaccuracyk1nn <- sum(diag(myaccuracytablek1nn))/sum(myaccuracytablek1nn)
print (mytestingaccuracyk1nn)