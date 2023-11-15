#Will Henshon, s2539250; ADD YOUR NAMES AND NUMBERS HERE
#
#Git address: https://github.com/MaxBoulanger/Practical-4
#
#



netup <- function(d){
  #Takes a vector of values, and initializes a neural network with nodes accord-
  #ing to the input vector. To do this, it creates a list of node vectors, a
  #list of weight matrices, and a list of offset vectors. The weight matrices
  #and offset vectors are given starting values according to a uniform random
  #distribution with the bounds (0,0.2).
  #
  #Inputs:
  # d - vector containing the shape of the neural network (i.e. the number
  #     of nodes at each level)
  #
  #Outputs:
  # nn - list containing a list of nodes, h, (all set to 0), a list of weight
  #      matrices, w, (initialized with random values) and a list of offset 
  #      vectors, b (initialized with random values)
  
  #Create a list for the nodes
  h <- list()
  
  #At each level of the network, generate a vector with the correct number of
  #nodes and store it in the h list
  for(i in 1:length(d)){
    h[[i]] = rep(0,d[i])
  }
  
  #Create lists to store the weight matrices and offset vectors
  w <- list()
  b <- list()
  
  #For each of the levels except the final one:
  for(i in 1:(length(d)-1)){
    #Create and store a weight matrix with rows equal to the size of the next
    #level and columns equal to the size of the current one, with values uniform
    #random on (0,0.2)
    w[[i]] <- matrix(runif(1,0,0.2), length(h[[i+1]]), length(h[[i]] ) )
    #Create and store an offset vector of size equal to the next level, with
    #values uniform random on (0,0.2)
    b[[i]] <- runif(length(h[[i+1]]), 0, 0.2) #Make sure these are actually random
  }
  
  #Create and return the list
  nn <- list(h=h, w=w, b=b)
  return(nn)
}

forward <- function(nn, inp){
  #Takes a network list (containing nodes, weight matrices, and offset vectors)
  #and an input vector (assigned to the first level of nodes) and calculates
  #the remaining nodes using the weight matrices and offset vectors provided.
  #
  #Inputs
  # nn - network list containing at least h, a list of node vectors, w, a list
  #     of weight matrices, and b, a list of offset vectors, all with the 
  #     correct sizes for the shape of the network
  # inp - input vector of size equal to the number of nodes in the first level
  #
  #Outputs
  # h - list of node vectors containing the calculated node values using the
  #     input vector and the model matrices
  
  #Retrieve h for ease of use
  h<- nn$h
  
  
  #Fill in the first node level with the input vector
  h[[1]] <- inp
  
  #I BET WE CAN VECTORIZE THIS
  #Fill in the remaining values for the nodes using the input vector
  for (i in 1:(length(h)-1)){
    #Calculate w*h + b for the given level
    vec <- (nn$w[[i]] %*% h[[i]]) + nn$b[[i]]
    #Use a helper function to only include values greater than 0
    vec2 <- sapply(vec, h_val)
    #Store the new node vector
    h[[i+1]] <- vec2
  }
  
  #Return the list of node vectors
  return(h)
}

h_val <- function(val){
  #Helper function for forward. Takes a value and returns
  #the value if greater than 0, and 0 otherwise.
  #Inputs:
  # vec - a numerical value
  #Outputs
  # to_return - returns 0 if the inputted value is less than 0 and the inputted
  #             value otherwise
  
  #Check if val is less than 0; return 0 if so
  if(val<=0){
    return(0)
  }
  #Otherwise return val
  else{
    return(val)
  }
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
  vec[iris$Species=='setosa'] <- 1
  vec[iris$Species=='versicolor'] <- 2
  vec[iris$Species=='virginica'] <- 3
  
  output_indices = (1:30)*5
  iris_train <- iris[-output_indices, ]
  vec_train <- vec[-output_indices]
  
  
  #Modify iris so that it only includes 4 of every 5 rows
  nn1<- train(nn, iris_train, vec_train)
  
  iris_predict <- iris[output_indices,]
  for(i in 1:length(iris_predict[,1])){
    iris_vec <- c(iris_predict[i,1],iris_predict[i,2],iris_predict[i,3],iris_predict[i,4])
    nn_temp <- forward(nn1, iris_vec)
    print(nn_temp[[4]])
  }
  
  
}

irisFunct()


#I fixed up the indexing bugs we were having, and added some prediction code. Seems like the values are bad, unless I'm misinterpreting them.
#To do: vectorize what we can to speed things up and avoid too many for loops. Figure out why the values are bad (probably take his advice of trying finite differencing), get the model trained,
# and add comments (I started)