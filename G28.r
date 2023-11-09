#Will Henshon, s2539250; ADD YOUR NAMES AND NUMBERS HERE
#
#Git address: https://github.com/MaxBoulanger/Practical-4
#
#



netup <- function(d){
  h <- list()
  for(i in 1:length(d)){
    h[[i]] = rep(0,d[i])
  }
  
  w <- list()
  b <- list()
  for(i in 1:length(d)-1){
    w[[i]] <- matrix(runif(1,0,0.2), length(h[[i+1]]), length(h[[i]] ) )
    b[[i]] <- runif(length(h[[i+1]]), 0, 0.2) #Make sure these are actually random
  }
  
  

  nn <- list(h=h, w=w, b=b)
  return(nn)
}

forward <- function(nn, inp){
  
  h<- nn$h
  
  for (i in 1:length(h[[1]])){
    h[[1]][i] <- inp[i]
  }
  
  for (i in 1:length(h)-1){
    vec <- (nn$w %*% h[[i]]) + nn$b
    vec2 <- sapply(vec, h_val)
    h[[i+1]] <- vec2
  }
  
  return(h)
}

h_val <- function(vec){
  to_retun <- max(0,vec)
}

backward <- function(nn, k){
  L = length(nn$h)
  h <- nn$h
  
  dh <- list()
  db<- list()
  dW <- list()
  for(j in 1:h[[L]]){
    dh[[L]]<-rep(0,j)
    if(j!=k){
      dh[[L]][j] <- exp(h[[L]][j])/sum(h[[L]])
    }
    else{
      dh[[L]][j] <- (exp(h[[L]][j])/sum(h[[L]]))-1
    }
  }
  
  for(i in (L-1):1){
    d<-rep(0,length(h[[i]]))
    for(j in 1:length(h[[i]])){
      if(h[[i+1]][j]>0){
        d[j] <- dh[[i+1]][j]
      }
      else{
        d[j]<-0
      }
    }
    dh[[i]] <- t(nn$w[[i]]) %*% d
    db[[i]] <- d
    dW[[i]] <- d%*%t(h[[i]])
  }
  
  nn[[dh]] <- dh
  nn[[db]] <- db
  nn[[dW]] <- dW
  
  return(nn)
}

train <- function(nn, inp, k, eta=0.01, mb = 10, nstep = 10000){
  
  
  
  for(m in 1:nstep){
    #Sample 10 points from the input
    data_sample <- sample(inp, mb)
    nn_list <- list()
    
    dh_average <- list()
    dW_average <- list()
    db_average <- list()
    
    #Run forward/backward on each point
    for(n in 1:mb){
      nn$h <- forward(nn, c(data_sample$Sepal.Length, data_sample$sepal.width,
                            data_sample$Petal.Length, data_sample$Petal.Width))
      nn_list[[n]] <- backward(nn,k)
      
      
      for(l in length(nn_list[[n]])){
        dh_average[[n]][l] <- dh_average[[n]][l] + nn_list[[dh]][l]
        dW_average[[n]][l] <- dW_average[[n]][l] + nn_list[[dW]][l]
        db_average[[n]][l] <- db_average[[n]][l] + nn_list[[db]][l]
      }
      
      
      
    }
    #Take the average of our 10 gradients
    for(l in length(dh_average[[1]])){

      dh_average[[n]][l] <- dh_average[[n]][l]/mb
      dW_average[[n]][l] <- dW_average[[n]][l]/mb
      db_average[[n]][l] <- db_average[[n]][l]/mb
    }
    #Get new gradients; store these
    for(i in 1:length(nn$h)){
      nn$w[[i]] <- nn$w[[i]] - (eta*dW_average[[i]])
      nn$b[[i]] <- nn$b[[i]] - (eta*db_average[[i]])
    }
  }
  
  return(nn)
}

irisFunct <- function(){
  netup(c(48))
}





