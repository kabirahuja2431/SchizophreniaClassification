set.seed(0)

#forward prop to Full connected layer
affine_forward <- function(input,W,b){
  z = sweep(input%*%W,2,-b)
  return(z)
}

#Back prop through FC layer
affine_backward <- function(input,W,b,dout){
  m = dim(input)[1]
  dW = (t(input)%*%dout)/m
  db = colSums(dout)/m
  dinput = dout%*%t(W)
  grads = list(dW,db,dinput)
  return(grads)
}
#Relu activation function forward pass
relu_forward <- function(z){
  z[z<0] = 0
  return(z)
}
#Relu activation function backward pass
relu_backward <- function(z,dout){
  dout[z<0] = 0
  return(dout)
}
#sigmoid activation function forward pass
sigmoid_forward <- function(z){
  a = 1/(1+exp(-z))
}
#Loss Function
loss_func <- function(y,a,m){
    loss = (-1/m)*(sum(y*log(a) + (1-y)*log(1-a)))
    return(loss)
  }

two_l_nn_model <- function(X,y,h = 50){
  sample = sample.int(n = nrow(X), size = floor(.6*nrow(X)), replace = F)
  X_train = X[sample,]
  X_test = X[-sample,]
  y_train = y[sample,]
  y_test = y[-sample,]

  m = dim(X_train)[1]
  n = dim(X_train)[2]
  #Weights and bias for input to hidden layer
  W1 = matrix(rnorm(n*h,0,1),n,h)*sqrt(2/(n))
  b1 = array(0,h)
  #Weights and bias for hidden layer to output
  W2 = matrix(rnorm(n*h,0,1),h,1)*sqrt(2/(n))
  b2 = 0
  #Gradient Descent
  lr = 0.01
  lamda = 0.5
  losses = c()
  for(i in 1:1000){
    #forward pass
    z1 = affine_forward(X_train,W1,b1)
    a1 = relu_forward(z1)
    z2 = affine_forward(a1,W2,b2)
    a2 = sigmoid_forward(z2)
    #backward pass
    dz2 = matrix((a2 - y_train),length(y_train),1)
    grads = affine_backward(z1,W2,b2,dz2)
    dW2 = matrix(unlist(grads[1]),h,1) + lamda*W2/m
    db2 = array(unlist(grads[2],1))
    dz1 = matrix(unlist(grads[3]),m,h)
    grads = affine_backward(X_train,W1,b1,dz1)
    dW1 = matrix(unlist(grads[1]),n,h)+lamda*W1/m
    db1 = array(unlist(grads[2]),h)
    #Updates
    
    W1 = W1 - (lr)*(dW1+lamda*W1/m)
    b1 = b1 - (lr)*db1
    W2 = W2 - (lr)*(dW2 + lamda*W2/m)
    b2 = b2 - (lr)*db2
    loss = loss_func(y_train,a2,m)
    reg_loss = (1/(2*m))*(sum(W1^2) + sum(W2^2))
    loss = loss + reg_loss
    losses = c(losses,loss)
  }
  plot(losses,type='l')
  #Testing Time
  z1 = affine_forward(X_test,W1,b1)
  a1 = relu_forward(z1)
  z2 = affine_forward(a1,W2,b2)
  a2 = sigmoid_forward(z2)
  a2[a2>0.5] = 1
  a2[a2<=0.5] = 0
  test_accuracy = sum(a2==y_test)/(length(y_test)) 
  print(test_accuracy)
  
  
}




#Reading the data

#SBM Data
X1 <- read.csv('data/train_SBM.csv')
X1 <- as.matrix(X1)
m = dim(X1)[1]
n = dim(X1)[2]-1
X1 = X1[1:m,2:(n+1)]
colnames(X1) <- NULL
y = read.csv('data/train_labels.csv')
y = array(y[1:m,2],c(m,1))

#FNC Data
X2 <- read.csv('data/train_FNC.csv')
X2 <- as.matrix(X2)
m = dim(X2)[1]
n = dim(X2)[2]-1
X2 = X2[1:m,2:(n+1)]
colnames(X2) <- NULL

#Taking both SBM and FNC data
X <- cbind(X1,X2)
m = dim(X)[1]
n = dim(X)[2]
colnames(X) <- NULL
for(h in 30:40){
  print(h)
  two_l_nn_model(X,y,h)
}