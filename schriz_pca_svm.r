library('caret')
set.seed(0)

#Function for the SVM model
#Takes the Data Frame as the arguement
svm_model <- function (train_data,test_data,method='svmLinear'){

  train_data[,ncol(train_data)] = factor(train_data[,ncol(train_data)])
  #Building SVM Model
  ctrl = trainControl(method="repeatedcv",number=5,repeats=1)
  grid = expand.grid(C = c(0.01,0.05,0.1,0.25,0.4,0.45,0.5,0.55,0.6,0.75,0.9,1,1.25,1.5,1.75,2,5))
  svm = train(x=train_data[,1:(ncol(train_data)-1)],y=train_data[,ncol(train_data)],method=method,
                     trControl=ctrl,preProcess = c("center", "scale"),
                     #tuneGrid = grid,
                     tuneLength = 10)
  svm
  #conf_mat = confusionMatrix(test_preds,test_data[,ncol(test_data)])
  
}

#Reading the data
set.seed(0)
X1 <- read.csv('data/train_FNC.csv')
X1 <- as.matrix(X1)
m = dim(X1)[1]
n = dim(X1)[2]-1
X1 = X1[1:m,2:(n+1)]
X2 <- read.csv('data/train_SBM.csv')
X2 <- as.matrix(X2)
m = dim(X2)[1]
n = dim(X2)[2]-1
X2 = X2[1:m,2:(n+1)]
X <- cbind(X1,X2)
m = dim(X)[1]
n = dim(X)[2]
colnames(X) <- NULL
y = read.csv('data/train_labels.csv')
y = array(y[1:86,2],c(86,1))
#train_test split
sample = sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = F)
X_train = X[sample,]
X_test = X[-sample,]
y_train = y[sample,]
y_test = y[-sample,]
#Mean Normalization
means = colMeans(X_train)
X_train = sweep(X_train,2,means)
X_test = sweep(X_test,2,means)

m = dim(X_train)[1]
#Performing PCA
sigma = (t(X_train) %*% X_train)/m
a = svd(sigma)
d = unlist(a[1])
U = matrix(unlist(a[2]),n)

#training the model by using 20-60 principle componenets and comparing the performance on test data
train_accuracies = c()
test_accuracies = c()
for (k in 10:60){
  x_train = X_train %*% U[1:n,1:k]
  x_test = X_test %*% U[1:n,1:k]
  train = cbind(x_train,y_train)
  train = data.frame(train)
  test = cbind(x_test,y_test)
  test = data.frame(test)
  svm = svm_model(train,test,'svmLinear')
  train_preds = predict(svm,newdata=train[,1:(ncol(test)-1)])
  train_accuracy = sum(train_preds==y_train)/length((y_train))
  test_preds = predict(svm, newdata=test[,1:(ncol(test)-1)])
  test_accuracy = sum(test_preds==y_test)/length(y_test)
  print(paste("For first",k,"Principle Components:"))
  print(paste("Train Accuracy",train_accuracy))
  print(paste("Test Accuracy",test_accuracy)) 
  test_accuracies = c(test_accuracies,test_accuracy)
}
#Selecting the number of components which gave max accuracy on test set
k = which.max(test_accuracies) + 10 -1

