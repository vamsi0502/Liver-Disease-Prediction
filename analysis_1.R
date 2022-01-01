
##################### Data Analysis #####################################

## Analysis of Indian Liver Patient Dataset: Classification Problem ##

rm(list = ls())

library(dplyr)
library(caret)
library(corrplot)
library(randomForest)
library(glmnet)
library(e1071)
library(xgboost)

setwd("C:/Users/vamsi/Desktop/UNL/Pet_Projects/indian_liver-patient")
data = read.csv("indian_liver_patient_train.csv")
#Caterogry 1 corresponds to Disease and 2 to No Disease. Renaming to 0 for No Disease
data$Category[which(data$Category==2)] = 0
#train.data$Category = as.factor(train.data$Category) #Converting response to a factor variable

na.obs = apply(data, 2, function(x) sum(is.na(x))) #Checking if there are any NAs in the dataset
na.to.rem = which(is.na(data$AGR)) #Only Albumin_and_Globulin_Ratio has 3 NAs.
data = data[-na.to.rem,] 

## Creating training and validation sets 80/20 split
train_index = sample(1:nrow(data), 0.8*nrow(data)) 
val_index = setdiff(1:nrow(data), train_index)

train.data = data[train_index,!colnames(data) %in% c("ID")]
val.data = data[val_index,!colnames(data) %in% c("ID")]


y = train.data$Category
X = train.data[,!colnames(train.data) %in% c("ID", "Category")]

#Looking at correlations between cts predictors
X.cor = cor(X[, !colnames(X) %in% c("Gender")])
X.cor


corrplot.mixed(X.cor)

# Total_Bil and Dir_bil have high corr, Ala and Asp have high corr, tot_pro and Alb have high cor, Alb and AGR have high corr. 

#### Classification Models  ####

accuracy_table = c()

# Logistic Regression
log_reg1 = glm(Category ~ ., data = train.data, family = "binomial")
summary(log_reg1)

logReg_yHat = predict.glm(log_reg1, newdata = val.data, type = "response")
logReg_pred_classes = c(ifelse(logReg_yHat>0.5, 1, 0))
names(logReg_pred_classes) = c()

obs_classes = val.data$Category

confusionMatrix(data = factor(logReg_pred_classes), reference = factor(obs_classes), positive = "1")

log_confusion_table = table(obs_classes, logReg_pred_classes)

table(obs_classes, pred_classes) %>% 
  prop.table() %>% round(digits = 3)

log_acc = (log_confusion_table[1,1]+log_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("logReg", log_acc))

# Random Forests

rf1 = randomForest(factor(Category) ~ ., data = train.data)

rf_yHat = predict(rf1, newdata = val.data, type = "response")
rf_pred_classes = rf_yHat
names(rf_pred_classes) = c()

obs_classes = val.data$Category


rf_confusion_table = table(obs_classes, rf_pred_classes)
rf_confusion_table

rf_acc = (rf_confusion_table[1,1]+rf_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("RF", rf_acc))

# Penalized logistic regression

set.seed(123)

X = model.matrix(Category ~ ., data = train.data)[,-1]
y = factor(train.data$Category)

#LASSO
cv.plr1 = cv.glmnet(x = X, y = factor(y), family = "binomial", alpha = 1)
plr1 = glmnet(x = X, y = factor(y), family = "binomial", alpha = 1, lambda = cv.plr1$lambda.min)

X.val = model.matrix(Category ~ ., data = val.data)[,-1]
plr1_yHat = predict(object = plr1, newx = X.val, type = "response")  
plr1_pred_classes = ifelse(plr1_yHat>0.5, 1, 0)
names(plr1_pred_classes) = c()

obs_classes = val.data$Category


plr1_confusion_table = table(obs_classes, plr1_pred_classes)
plr1_confusion_table

lasso_acc = (plr1_confusion_table[1,1]+plr1_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("LASSO", lasso_acc))

#Elastic Net
cv.plr2 = train(factor(Category) ~ ., data = train.data, method = "glmnet", 
             trControl = trainControl("cv", number = 10), tuneLength = 10)
cv.plr2$bestTune

plr2 = glmnet(x = X, y = y, family = "binomial", alpha = cv.plr2$bestTune[1],
              lambda = cv.plr2$bestTune[2])


X.val = model.matrix(Category ~ ., data = val.data)[,-1]
plr2_yHat = predict(object = plr2, newx = X.val, type = "response")  
plr2_pred_classes = ifelse(plr2_yHat>0.5, 1, 0)
names(plr2_pred_classes) = c()

obs_classes = val.data$Category


plr2_confusion_table = table(obs_classes, plr2_pred_classes)
plr2_confusion_table

en_acc = (plr2_confusion_table[1,1]+plr2_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("ENET", en_acc))

#Ridge Regression
cv.plr3 = cv.glmnet(x = X, y = factor(y), family = "binomial", alpha = 0)
plr3 = glmnet(x = X, y = factor(y), family = "binomial", alpha = 0, lambda = cv.plr1$lambda.min)

X.val = model.matrix(Category ~ ., data = val.data)[,-1]
plr3_yHat = predict(object = plr3, newx = X.val, type = "response")  
plr3_pred_classes = ifelse(plr3_yHat>0.5, 1, 0)
names(plr3_pred_classes) = c()

obs_classes = val.data$Category


plr3_confusion_table = table(obs_classes, plr3_pred_classes)
plr3_confusion_table


rr_acc = (plr3_confusion_table[1,1]+plr3_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("RidgeReg", rr_acc))

# Support Vector Machines

dat = data.frame(x = train.data[,!colnames(train.data) %in% c("ID", "Category")], y = train.data$Category)
val.dat = data.frame(x = val.data[,!colnames(train.data) %in% c("ID", "Category")], y = val.data$Category)
svm1 = svm(factor(y) ~ ., data = dat, kernel = "linear", cost = 10, scale = F)

plot(svm1, dat)
svm_yHat = predict(object = svm1, newdata = val.dat)  
svm_pred_classes = svm_yHat
names(svm_pred_classes) = c()

obs_classes = val.data$Category


svm_confusion_table = table(obs_classes, svm_pred_classes)
svm_confusion_table

svm_acc = (svm_confusion_table[1,1]+svm_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("SVM", svm_acc))

## xgboost
dat = train.data[,!colnames(train.data) %in% c("ID", "Category")]
dat = xgb.DMatrix(as.matrix(sapply(dat,  as.numeric)),  label = train.data$Category)

xgb1 = xgboost(dat,
               max_depth = 6, eta = 1, nthread = 2, nrounds = 100, objective = "binary:logistic")

val.dat = val.data[,!colnames(train.data) %in% c("ID", "Category")]
val.dat = xgb.DMatrix(as.matrix(sapply(val.dat, as.numeric)), label = val.data$Category)
xgb_yHat = predict(xgb1, val.dat)

xgb_yHat1 = as.numeric(xgb_yHat > 0.5)


watchlist = list(train = dat, test = val.dat)

xgb2 = xgb.train(data = dat, max_depth = 6, eta = 0.1, nthread = 2, nrounds = 10, watchlist = watchlist, 
                 objective = "binary:logistic")

xgb_pred = predict(xgb2, val.dat)
xgb_pred1 = as.numeric(xgb_pred > 0.5)

obs_classes = val.data$Category
xgb_confusion_table = table(obs_classes, xgb_pred1)
xgb_confusion_table


xgb_acc = (xgb_confusion_table[1,1]+xgb_confusion_table[2,2])/nrow(val.data)
accuracy_table = rbind(accuracy_table, c("XGBoost", xgb_acc))


