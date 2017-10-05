library(rpart)
#source('schriz_logreg.r')
set.seed(0)
      
      decision_tree<-function(X,y){
        sample = sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = F)
        X_train = X[sample,]
        y_train=y[sample,]
        X_test = X[-sample,]
        y_test=y[-sample,]
        data1=cbind(X_train,y_train[2])
        form<-reformulate( names(X_train),response=names(y_train))
        fit<-rpart(form, data=data1, method="class")
        Prediction<- predict(fit, X_test, type = "class")
        accuracy=sum(Prediction==y_test$Class)/length(y_test$Class)
        print(accuracy)
      }
      
      
      #SBM Data
      X1 <- read.csv('data/train_SBM.csv')
      X1 <- X1[1:nrow(X1),2:ncol(X1)]

      y = read.csv('data/train_labels.csv')
      y = array(y[1:86,2],c(86,1))      
      #FNC Data
      X2 <- read.csv('data/train_FNC.csv')
      X2 <- X2[1:nrow(X2),2:ncol(X2)]
      
      
      #Taking both SBM and FNC data
      X_full <- cbind(X1,X2)
      
      decision_tree(X1,y)
      decision_tree(X2,y)
      decision_tree(X_full,y)