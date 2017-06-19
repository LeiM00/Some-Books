## Chapter 3. Linear Regression

library(MASS)
library(ISLR)
library(car)

names(Boston)

attach(Boston)
lm.fit = lm(medv~lstat+age+log(rm), data = Boston)
lm.fit = lm(medv~lstat*age, data = Boston)  # lstat+age+lstat:age
lm.fit = lm(medv~lstat+I(lstat^2))
lm.fit5=lm(medv~poly(lstat ,5))
lm.fit = lm(medv~., data = Boston)
lm.fit1=lm(medv~.-age,data = Boston)
summary(lm.fit)
names(lm.fit)
coef(lm.fit)  # lm.fit$coefficients
confint(lm.fit)

predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "confidence")    # interval = "prediction"

plot(lstat, medv, col = "red", pch = 20)
abline(lm.fit, lwd = 3, col = "red")

par(mfrow = c(2,2))
plot(lm.fit)
plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))
plot(hatvalues(lm.fit))    # Leverage statistics
which.max(hatvalues(lm.fit))
summary(lm.fit)$r.sq       # ?summary.lm
vif(lm.fit)                # car library


## Chapter 4. Classification
dim(Smarket)
cor(Smarket[,-9])    # pairwise correlations
# Logistic regression function: glm() with option family = binomial
train = Smarket$Year<2005
glm.fit = glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data = Smarket, family = binomial, subset = train)
glm.probs = predict(glm.fit, Smarket[!train,], type = "response")
glm.pred = rep("Down", 252)
glm.pred[glm.probs>0.5] = "Up"
table(glm.pred, Smarket[!train,]$Direction)  # the table() function can be used to produce a confusion matrix
mean(glm.pred == Smarket[!train,]$Direction)
predict(glm.fit, newdata = data.frame(Lag1 = c(1.2, 1.5), Lag2 = c(1.1, -0.8)), type = "response")
# LDA: lda() function in MASS library
lda.fit = lda(Direction~Lag1+Lag2, data = Smarket, subset = train)
lda.pred = predict(lda.fit, Smarket[!train,])    # names(lda.pred)
table(lda.pred$class, Smarket[!train,]$Direction)
mean(lda.pred$class == Smarket[!train,]$Direction)
sum(lda.pred$posterior[,1]>=.5)
# QDA: qda() function in MASS library
qda.fit = qda(Direction~Lag1+Lag2, data = Smarket, subset = train)
qda.pred = predict(qda.fit, Smarket[!train,])    
table(qda.pred$class, Smarket[!train,]$Direction)
mean(qda.pred$class == Smarket[!train,]$Direction)
# K Nearest Neighbors
library(class)
train.X = cbind(Smarket$Lag1, Smarket$Lag2)[train,]
test.X = cbind(Smarket$Lag1, Smarket$Lag2)[!train,]
train.Direction = Smarket$Direction[train]
set.seed(1)
knn.pred = knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Smarket$Direction[!train])
mean(knn.pred == Smarket$Direction[!train])
# example
standardized.X = scale(Caravan[,-86])  # do not scale qualitative variable
test = 1:1000
train.X = standardized.X[-test,]
test.X = standardized.X[test,]
train.Y = Caravan$Purchase[-test]
test.Y = Caravan$Purchase[test]
knn.pred = knn(train.X, test.X, train.Y, k = 1)
mean(test.Y == knn.pred)

glm.fit=glm(Purchase~.,data=Caravan ,family=binomial, subset=-test)  # logistic regression does not need scaling?
glm.probs=predict(glm.fit,Caravan[test,],type="response")
glm.pred=rep("No",1000)
glm.pred[glm.probs >.5]="Yes"
table(glm.pred,test.Y)
glm.pred=rep("No",1000)
glm.pred[glm.probs >.25]=" Yes"
table(glm.pred,test.Y)


## Chapter 5. Resampling Methods
# Validation Set Approach
train = sample(392, 196)    # split the set of observations by selecting a random subset of 196 out of the original 392 observations
attach(Auto)
lm.fit = lm(mpg~horsepower, data = Auto, subset = train)
mean((mpg-predict(lm.fit, Auto))[-train]^2)
lm.fit2 = lm(mpg~poly(horsepower,2), data = Auto, subset = train)
mean((mpg-predict(lm.fit2, Auto))[-train]^2)
lm.fit3 = lm(mpg~poly(horsepower,3), data = Auto, subset = train)
mean((mpg-predict(lm.fit3, Auto))[-train]^2)
# Leave one out cross validation/ k fold cross validation
library(boot)
glm.fit = glm(mpg~horsepower, data = Auto)   # glm() without family argument performs linear regression just like the lm() function.
cv.err = cv.glm(Auto, glm.fit, K = 10)    # cv.glm function, default k = 1: LOOCV
cv.err$delta   # The two numbers in the delta vector contain the cross-validation results (standard k-fold CV estimate, a bias corrected version)
cv.error = rep(0,5)
for (i in 1:5){
  glm.fit = glm(mpg~poly(horsepower,i), data = Auto)
  cv.error[i] = cv.glm(Auto, glm.fit, K = 10)$delta[1]}
# Bootstrap
# First, create a function that computes the statistic of interest
# Second, use the boot() function, which is part of the boot library, to perform the bootstrap by repeatedly sampling observations from the data set with replacement.
alpha.fn=function(data,index){
  X=data$X[index]
  Y=data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))}
alpha.fn(Portfolio ,1:100)
alpha.fn(Portfolio,sample(100,100,replace=T))
boot(Portfolio ,alpha.fn,R=1000)
# Using bootstrap to estimate the accuracy of a linear regression model
boot.fn = function(data, index) 
  return (coef(lm(mpg~horsepower, data = data, subset = index)))
boot.fn(Portfolio, sample(392, 392, replace = T))
boot(Portfolio, boot.fn, R =1000)



## Linear Model Selection and Regularization
# Best Subset Selection
Hitters=na.omit(Hitters)    # na.omit(): removes all of the rows that have missing values in any variable.
# regsubsets() function (part of the leaps library) performs best subset selection by identifying the best model that contains a given number of predictors
regfit.full=regsubsets(Salary~.,data=Hitters ,nvmax=19)
reg.summary=summary(regfit.full)
names(reg.summary)
reg.summary$rsq
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS", type="l")
plot(reg.summary$adjr2 ,xlab="Number of Variables ", ylab="Adjusted RSq",type="l")
which.min(reg.summary$cp)
which.min(reg.summary$bic)
plot(regfit.full,scale="r2")   # "adjr2", "Cp", "bic"
coef(regfit.full ,6)

# Forward and Backward Stepwise Selection
regfit.fwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "forward")
regfit.bwd = regsubsets(Salary~., data = Hitters, nvmax = 19, method = "backward")

# Choosing Among Models Using the Validation Set Approach and Cross-Validation
set.seed(1)
train = sample(c(TRUE, FALSE), nrow(Hitters), rep = TRUE)
test = (!train)
regfit.best=regsubsets(Salary~.,data=Hitters[train,], nvmax =19)
test.mat=model.matrix(Salary~.,data=Hitters[test,])
val.errors = rep(NA, 19)
for (i in 1:19){
  coefi = coef(regfit.best, id = i)
  pred = test.mat[, names(coefi)]%*%coefi
  val.errors[i] = mean((Hitters$Salary[test] - pred)^2)
}

# ...



x1 = seq(0,1,0.01)
x2 = seq(0,1,0.01)
f = x1^2 - x1*x2 + x2^2 - 5*x1 - 2*x2 + 10
