#Will HENSHON, s2539250; ADD YOUR NAMES AND NUMBERS HERE
#Max BOULANGER, s2599704
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
    w[[i]] <- matrix(runif(length(h[[i+1]])*length(h[[i]]),0,0.2), length(h[[i+1]]), length(h[[i]] ) )
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
  #print('h')
  #print(h)
  
  #Return the list of node vectors
  return(h)
}

h_val <- function(val){
  #Helper function for forward. Takes a value and returns
  #the value if greater than 0, and 0 otherwise.
  #Inputs:
  # val - a numerical value
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
  #Takes a network list (containing nodes, weight matrices, and offset vectors)
  #and an output class and calculated the derivatives of the loss function
  #
  #Inputs
  # nn - network list returned from forward containing at least h, a list of node 
  #      vectors, w, a list of weight matrices, and b, a list of offset vectors, 
  #      all with the correct sizes for the shape of the network
  # k - output class that we are interested in
  #
  #Outputs
  # nn - netowrk list updated by adding the derivatives with regard to nodes, dh,
  #      weights, dW and offset, db
  
  #Find the number of layers
  L = length(nn$h)
  #Retrieve the nodes for easy access
  h <- nn$h
  
  print('k')
  print(k)
  
  print('h')
  print(h[[L]])
  
  print('Loss')
  l <- loss(k,h[[L]])
  print(l)
  
  
  
  #Initialize lists to store each of the derivatives
  dh <- list()
  db<- list()
  dW <- list()
  
  #Create a vector to store the final layer of derivatives
  dh[[L]]<-rep(0,length(h[[L]]))
  
  #Set the derivatives of loss with respect to the final layer of nodes
  dh[[L]] <- (exp(h[[L]])) /sum(exp(h[[L]]))
  dh[[L]][k] <- dh[[L]][k] - 1
  
  
  print('dh[L]')
  print(dh[[L]])
  
  # l<- loss(k,h[[L]],1)
  # dl <- finite_difference(k,h[[L]])
  # print(dl)
  # print(dh[[L]][k])
  
  #Iterate backwards across all layers except the last one
  for(i in (L-1):1){
    print('i'); print(i)
    d2 <- rep(0,length(h[[i+1]]))
    zeros <- which(dh[[i+1]]<0)
    d2 <- dh[[i+1]]
    d2 <- c(d2)
    d2[zeros] <- 0
    
    print('d2'); print(d2)
    
    print('h[i]');print(h[[i]])
    
    #print('d')
    #print(d2)
    
    #print('d2')
    #print(d2)

    #print('d')
    #print(d)

    
    #Compute the derivatives to be stored
    dh[[i]] <- t(nn$w[[i]]) %*% d2
    db[[i]] <- d2
    dW[[i]] <- d2%*%t(h[[i]])
    
    print('w')
    print(nn$w[[i]])
    
    print('dW')
    print(i)
    print(dW[[i]])
    
    #print('dW')
    #print(dW[[i]])
  }
  #print('dh')
  #print(dh)
  
  #print('w')
  #print(nn$w)
  
  #print('b')
  #print(nn$b)
  
  #Store the results to the nn model list and return
  nn[['dh']] <- dh
  nn[['db']] <- db
  nn[['dW']] <- dW
  return(nn)
}

train <- function(nn, inp, k, eta=0.01, mb = 10, nstep = 10000){
  #Takes a network list (containing nodes, weight matrices, and offset vectors),
  #an input vector and its corresponding labels, the step size, the number of 
  #data to randomly sample to compute the gradient and the number of 
  #optimization steps to take, and trains our network 
  #
  #Inputs
  # nn - network list returned from forward containing at least h, a list of node 
  #      vectors, w, a list of weight matrices, b, a list of offset vectors, and
  #      the derivatives with regard to nodes, dh, weights, dW and offset, db 
  #      all with the correct sizes for the shape of the network
  # inp - input vector of size equal to the number of nodes in the first level
  #
  # k - output class that we are interested in
  #
  # eta - step size used to update the parameters
  #
  # mb -  number of data to randomly sample to compute the gradient
  #
  # nstep - number of optimization steps to take
  #
  #Outputs
  #
  # nn - updated network list with new node vectors w and new list of offset
  #      vectors
  
  for(m in 1:1){
    #Sample mb points from the input
    data_sample <- sample(1:length(inp[,1]), mb,replace=TRUE)
    nn_list <- list()
    
    n=1
    nn$h<-forward(nn, c(inp[data_sample[n],1], inp[data_sample[n],2],
                        inp[data_sample[n],3], inp[data_sample[n],4]))
    ls <- loss(k[1],nn$h[[length(nn$h)]],1)
    w <- nn$w[[3]]
    nn$w[[3]][1,1] <- nn$w[[3]][1,1] + 10^(-7)
    nn$h2 <- forward(nn, c(inp[data_sample[n],1], inp[data_sample[n],2],
                           inp[data_sample[n],3], inp[data_sample[n],4]))
    ls2 <- loss(k[1],nn$h2[[length(nn$h2)]],1)
    der <- (ls2-ls)/10^(-7)
    nn<-backward(nn,k[1])
    print('Derivatives')
    print(nn$dW[[1]][1,1])
    print(der)
    
    
    #Initialize lists to store each of the averages
    dh_average <- list()
    dW_average <- list()
    db_average <- list()
    dh_list <- list()
    dW_list <- list()
    db_list <- list()
    
    #Run forward/backward on each point
    for(n in 1:1){
      nn$h <- forward(nn, c(inp[data_sample[n],1], inp[data_sample[n],2],
                            inp[data_sample[n],3], inp[data_sample[n],4]))
      kstar <- k[data_sample]
      nn_matrix <- backward(nn,kstar[n])
      print(nn_matrix$dh)
      dh_list[[n]] <- nn_matrix$dh
      dW_list[[n]] <- nn_matrix$dW
      db_list[[n]] <- nn_matrix$db
      
      for(l in 1:length(nn_matrix$dh)){
        if (n == 1){
          dh_average[[l]] <- nn_matrix$dh[[l]]
        }
        else{
          dh_average[[l]] <- dh_average[[l]]+nn_matrix$dh[[l]]
        }
      }
      for(l in 1:length(nn_matrix$dW)){
        if (n == 1){
          dW_average[[l]] <- nn_matrix$dW[[l]]
        }
        else{
          dW_average[[l]] <- dW_average[[l]] + nn_matrix$dW[[l]]
        }
      }
      for(l in 1:length(nn_matrix$db)){
        
        if (n == 1){
          db_average[[l]] <- nn_matrix$db[[l]]
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
  
  #Return the updated list
  return(nn)
}

irisFunct <- function(){
  #This function aims to apply our model to the iris database
  
  set.seed(0)
  
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
    #Convert these into probabilities using the equations on the sheet
    sum_exp <- sum(exp(nn_temp[[4]]))
    probs <- exp(nn_temp[[4]]) / sum_exp
    #print(probs)
  }
  
  
}

loss <- function(k, hL, n=1){
  sum1 <- sum(exp(hL))
  print('Sum1')
  print(sum1)
  s1 <- log(exp(hL[k])/(sum1))
  print('Sum2')
  s2 <- s1/n
  print(s2)
  #print('s2')
  #print(s2)
  s3<- -sum(s2)
  return(s3)
}

irisFunct()




finite_difference <- function(k, hL){
  epsilon <-0.05
  hLk_new <- hL[k]+epsilon
  hL_new <- hL
  hL_new[k] <- hLk_new
  l1 <- loss(k, hL_new,1)
  l2 <- loss(k, hL, 1 )
  der <- (l1-l2)/epsilon
}


#I spent some more time bug fixing. I caught some mistakes, but the last one I
#caught (lines 141-156, we allowed a lot of d_j values to be <0 when they shouldn't have) actually made the values for the last set of nodes all
#converge to 0, which is definitely wrong. I couldn't find the mistake, but I
#think finite differencing will help. The code is also still very slow -- I
#vectorized some of it, but I don't really know how to handle the parts with lists.
#It implies on the sheet that we can use matrix operations but I'm not sure how.
#Otherwise, only some of the code is commented, but we can do that at the end.