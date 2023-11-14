#Will Henshon, s2539250; ADD YOUR NAMES AND NUMBERS HERE
#
#Git address: https://github.com/MaxBoulanger/Practical-4
#
#



netup <- function(d){
  h <- list()
  for(i in 1:length(d)){
    h[[i]] = rep(0,d[i])
    #print(h[[i]])
  }
  
  w <- list()
  b <- list()
  for(i in 1:(length(d)-1)){
    w[[i]] <- matrix(runif(1,0,0.2), length(h[[i+1]]), length(h[[i]] ) )
    b[[i]] <- runif(length(h[[i+1]]), 0, 0.2) #Make sure these are actually random
  }
  
  

  nn <- list(h=h, w=w, b=b)
  return(nn)
}

forward <- function(nn, inp){
  
  h<- nn$h
  #print(h)
  for (i in 1:(length(h[[1]]))){
    h[[1]][i] <- inp[i]
  }
  #print(inp)
  #print(h)
  for (i in 1:(length(h)-1)){
    vec <- (nn$w[[i]] %*% h[[i]]) + nn$b[[i]]
    vec2 <- sapply(vec, h_val)
    h[[i+1]] <- vec2
  }
  #print(h)
  return(h)
}

h_val <- function(vec){
  to_retun <- max(0,vec)
}

backward <- function(nn, k){
  L = length(nn$h)
  h <- nn$h
  #print(h)
  dh <- list()
  db<- list()
  dW <- list()
  for(j in 1:length(h[[L]])){
    dh[[L]]<-rep(0,j)
    if(j!=k){
      dh[[L]][j] <- exp(h[[L]][j])/sum(h[[L]])
    }
    else{
      dh[[L]][j] <- (exp(h[[L]][j])/sum(h[[L]]))-1
    }
  }
  
  for(i in (L-1):1){
    d<-rep(0,length(h[[i+1]]))
    for(j in 1:length(h[[i+1]])){
      #print(h[[i+1]])
      if(h[[i+1]][j]>0){
        d[j] <- dh[[i+1]][j]
      }
      else{
        d[j]<-0
      }
    }
    #print(t(nn$w[[i]]))
    #print(d)
    dh[[i]] <- t(nn$w[[i]]) %*% d
    db[[i]] <- d
    dW[[i]] <- d%*%t(h[[i]])
  }
  
  nn[['dh']] <- dh
  nn[['db']] <- db
  nn[['dW']] <- dW
  #print(nn)
  return(nn)
}

train <- function(nn, inp, k, eta=0.01, mb = 10, nstep = 10000){
  
  
  
  for(m in 1:nstep){
    #Sample 10 points from the input
    data_sample <- sample(1:length(inp$Sepal.Length), mb) #Generalize for any data set
    nn_list <- list()
    
    dh_average <- list()
    dW_average <- list()
    db_average <- list()
    dh_list <- list()
    dW_list <- list()
    db_list <- list()
    
    #Run forward/backward on each point
    for(n in 1:mb){
      #Generalize this later
      nn$h <- forward(nn, c(inp$Sepal.Length[data_sample[n]], inp$Sepal.Width[data_sample[n]],
                            inp$Petal.Length[data_sample[n]], inp$Petal.Width[data_sample[n]]))
      #print(nn$h)
      kstar <- k[data_sample]
      #print(kstar)
      nn_matrix <- backward(nn,kstar[n])
      dh_list[[n]] <- nn_matrix$dh
      dW_list[[n]] <- nn_matrix$dW
      db_list[[n]] <- nn_matrix$db
      #print(dh_list)
      #print(dW_list)
      #print(db_list)
      
      
      for(l in 1:length(nn_matrix$dh)){
        #print(nn_matrix$dh[[l]])
        if (n == 1){
          dh_average[[l]] <- nn_matrix$dh[[l]]
        }
        else{
        #print(dh_list[[l]][n])
          #dh_average[[l]] <- dh_average[[l]] + nn_matrix$dh[[l]]
        }
      }
      #print(dh_average)
      for(l in 1:length(nn_matrix$dW)){
        #print(nn_matrix$dW[[l]])
        if (n == 1){
          dW_average[[l]] <- nn_matrix$dW[[l]]
        }
        else{
          #dW_average[[l]] <- dW_average[[l]] + nn_matrix$dW[[l]]
        }
      }
      
      for(l in 1:length(nn_matrix$db)){
        
        if (n == 1){
          db_average[[l]] <- nn_matrix$db[[l]]
          #print(nn_matrix$db[[l]])
        }
        else {
          db_average[[l]] <- db_average[[l]] + nn_matrix$db[[l]]
        }
      }
      
    }
    #Take the average of our 10 gradients
    for(l in 1:length(dh_average[[1]])){
      dh_average[[l]] <- dh_average[[l]]/mb
    }
    for(l in 1:length(db_average)){
      dW_average[[l]] <- dW_average[[l]]/mb
      db_average[[l]] <- db_average[[l]]/mb
    }
    
    #Get new gradients; store these
    for(i in 1:(length(nn$h)-1)){
      nn$w[[i]] <- nn$w[[i]] - (eta*dW_average[[i]])
      nn$b[[i]] <- nn$b[[i]] - (eta*db_average[[i]])
    }
  }
  
  return(nn)
}

irisFunct <- function(){
  nn<- netup(c(4,8,7,3))
  vec = rep(0,length(iris$Species))
  vec[iris$Species=='Setosa'] <- 1
  vec[iris$Species=='versicolor'] <- 2
  vec[iris$Species=='virginica'] <- 3
  
  
  
  #Modify iris so that it only includes 4 of every 5 rows
  nn1<- train(nn, iris, vec)
  
  
}

irisFunct()


# Addition is adding lists as columns, matrix addition not happening
# Problem with dh values 


