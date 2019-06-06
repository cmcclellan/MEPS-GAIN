GAIN <- function(data, covariates=NULL, batch.size = 128, missing.rate=.2, 
                 hint.rate=.9, alpha=10, sum_equal_amt = 0, sum_less_chrg = 0, train.rate=.8, iterations = 50) {
  require(tensorflow, quietly = T)
  require(dplyr, quietly = T)

  set.seed(8675309)
  ###Paramaters####
  #Mini Batch Size
  mb_size <- batch.size
  #Missing Rate
  p_miss <- missing.rate
  #Hint Rate
  p_hint <- hint.rate
  #Loss Hyperparameter - RMSE
  alpha <- alpha
  #Loss Hyperparameter - total expend equality:  The amounts in
  #the first 10 outcomes (amt1-amt10) must sum to equal the 11th outcome (sumpay).
  #This is a hyperparameter for a modified loss function including this.
  beta <- sum_equal_amt
  #loss Hyperparameter - Sumpay<charge:  The amounts in
  #the 11th must be less than or equal to the 12th outcome (tlchrg).
  #This is a hyperparameter for a modified loss function including this.
  sigma <- sum_less_chrg
  #Train Rate
  train_rate <- train.rate
  
  #Hack to help explore if including covariates are useful or not
  #No covariates break the code later, so if no covariates are provided
  #include a non-informative vector of 1s to make the code run, but not
  #affect output.
  if (covariates %>% is.null()){
    covariates <- matrix(1, nrow(data)) %>% as.data.frame()
  }
  
  No <- nrow(data)
  Dim <- length(data)
  Dim.covars <- length(covariates)
  Missing <- matrix(runif(No*Dim,0,1)>p_miss, No, Dim)*1
  
  ## Initial pass at including covariates: combine imputation data
  ## and covariates with missing mask where all covariate are marked non-missing.
  ## Makes the GAN predict/classify ALL variables missing or not.  Alternative
  ## approach is to pass data/missing mask and covariates to GAN separately, combine in generator/discriminator,
  ## and keep output diminsions to only data dimensions.
  # if(covariates %>% is.null()){
  #   Dim <- length(data)
  #   #Create missing pattern
  #   Missing <- matrix(runif(No*Dim,0,1)>p_miss, No, Dim)*1
  # } else {
  #   Dim <- length(data)+length(covariates)
  #   
  #   Missing <- matrix(runif(No*length(data),0,1)>p_miss, No, length(data))*1
  #   covars.miss <- matrix(TRUE, No, length(covariates))*1
  #   Missing <- cbind(Missing, covars.miss)
  #   
  #   data <- cbind(data, covariates)
  #   
  # }
  
  
  #Hidden State dimensions - add covariate dimensions to hidden layers
  H_Dim1 <- Dim + Dim.covars
  H_Dim2 <- Dim  + Dim.covars
  
  #Normalize Data to 0-1 - Decimalize to closer maintain relationships b/t variables?
  # inflation_factors <- data %>% summarise_all((funs(log10(max(.)) %>% ceiling)))
  # data <- data %>% mutate_all(funs((.)/10^(log10(max(.)) %>% ceiling)))
  # covariates <- covariates %>%  mutate_all(funs((.)/10^(log10(max(.)) %>% ceiling)))

  
  inflation_factors <- data %>% summarise_all((funs((max(.)))))
  data <- data %>% mutate_all(funs(((.)-min(.))/(max(.)+1*10^-8)))


  
  #Split data to train/test
  idx <- sample(1:No, No*train_rate)
  trainX <-  data[idx,] %>% as.data.frame()
  trainM <-  Missing[idx,] %>% as.data.frame()
  trainC <- covariates[idx,] %>% as.data.frame()
  
  testX <- data[-idx,] %>% as.data.frame()
  testM <- Missing[-idx,] %>% as.data.frame()
  testC <- covariates[-idx,] %>% as.data.frame()
  
  #Input Placeholders####
  #Data vector
  X <- tf$placeholder(tf$float32, shape(NULL,Dim))
  #Mask Vector
  M <- tf$placeholder(tf$float32, shape(NULL,Dim))
  #Hint Vector
  H <- tf$placeholder(tf$float32, shape(NULL,Dim))
  # Data vector with missing values
  New_X <- tf$placeholder(tf$float32, shape(NULL,Dim))
  Covar_X <- tf$placeholder(tf$float32, shape(NULL,Dim.covars))
  
  
  #Discriminator Variables####
  xavier.int <- function(size) {
    in.dim <- size[1]
    x.std <- 1/tf$sqrt(in.dim/2)
    return(tf$random_normal(shape = size %>% as.integer(), stddev = x.std))
  }
  
  sample.M <- function(m, n, p) {
    mat.m <- runif(m*n, 0,1) %>% matrix(m)>p 
    mat.m = mat.m*1 
    return(mat.m) 
  }
  
  D_W1 <- tf$Variable(xavier.int(c((Dim*2)+Dim.covars, H_Dim1)))
  D_b1 <- tf$Variable(tf$zeros(H_Dim1))
  
  D_W2 <- tf$Variable(xavier.int(c(H_Dim1, H_Dim2)))
  D_b2 <- tf$Variable(tf$zeros(H_Dim2))
  
  D_W3 <- tf$Variable(xavier.int(c(H_Dim2, Dim)))
  D_b3 <- tf$Variable(tf$zeros(Dim) )
  
  theta_D <- c(D_W1,D_b1,D_W2,D_b2,D_W3,D_b3)
  
  #Generator Variables####
  G_W1 <- tf$Variable(xavier.int(c((Dim*2)+Dim.covars, H_Dim1)))
  G_b1 <- tf$Variable(tf$zeros(H_Dim1))
  
  G_W2 <- tf$Variable(xavier.int(c(H_Dim1, H_Dim2)))
  G_b2 <- tf$Variable(tf$zeros(H_Dim2))
  
  G_W3 <- tf$Variable(xavier.int(c(H_Dim2, Dim)))
  G_b3 <- tf$Variable(tf$zeros(Dim) )
  
  theta_G <- c(G_W1,G_b1,G_W2,G_b2,G_W3,G_b3)
  
  #####Gain Function####
  
  generator <- function(new_x, covars, m){
    inputs = tf$concat(axis=as.integer(1), values = c(new_x, covars, m))
    G_h1 = tf$nn$relu(tf$matmul(inputs, G_W1)+G_b1)
    G_h2 = tf$nn$relu(tf$matmul(G_h1, G_W2)+G_b2)
    G_prob = tf$nn$sigmoid(tf$matmul(G_h2, G_W3)+G_b3)
    return(G_prob)
  }
  
  discriminator <- function(new_x, covars, h){
    inputs = tf$concat(axis=as.integer(1), values = c(new_x, covars, h))
    D_h1 = tf$nn$relu(tf$matmul(inputs, D_W1)+D_b1 )
    D_h2 = tf$nn$relu(tf$matmul(D_h1, D_W2)+D_b2)
    D_logit = tf$matmul(D_h2, D_W3)+D_b3
    D_prob =   tf$nn$sigmoid(D_logit)
    return(D_prob)
  }
  
  ####Additional functions####
  sample.Z <- function(m,n){
    runif(m*n,0,.01) %>% matrix(m) %>% return()
    #matrix(1,m,n) %>% return()
  }
  
  #Structure####
  
  G_sample  <-  generator(New_X, Covar_X, M)
  
  Hat_New_X <-  New_X*M+G_sample*(1-M)
  
  D_prob <-  discriminator(Hat_New_X, Covar_X, H)
  
  #Loss####
  D_loss1 <- -tf$reduce_mean(M*tf$log(D_prob+1e-8) + (1-M)* tf$log(1-D_prob+1e-8))
  G_loss1 <- -tf$reduce_mean((1-M)*tf$log(D_prob + 1e-8))
  MSE_train_loss <- tf$reduce_mean((M*New_X-M*G_sample)^2)/tf$reduce_mean(M)
  
  #If trying to impute all the amounts (ie amt1-amt10, sumpay and tlchrg), 
  #add to the loss function the expenditure equality and sumpay/charge inequality.
  #Else, if only trying to impute sumpay, original loss function
  if (Dim>1){
    Pay_vect <- tf$concat(axis=1L, values=c(tf$slice(Hat_New_X, c(0L,0L), c(-1L,10L)), tf$scalar_mul(-1, tf$slice(Hat_New_X, c(0L,10L), c(-1L, 1L)))))
    Expend_equailty_loss <- tf$reduce_mean(tf$abs(tf$reduce_sum(Pay_vect, 1L)))
     
    Diff_Charge_Pay_a <- tf$minimum(tf$slice(Hat_New_X, c(0L, 11L), c(-1L, 1L)) - tf$slice(Hat_New_X, c(0L, 10L), c(-1L, 1L)),0)
    Diff_Charge_Pay <- -tf$reduce_mean(Diff_Charge_Pay_a)
     
    D_loss <- D_loss1 + beta * Expend_equailty_loss + sigma*Diff_Charge_Pay
    G_loss <- G_loss1+ alpha * MSE_train_loss 
  } else {
    D_loss <- D_loss1 
    G_loss <- G_loss1 + alpha * MSE_train_loss 
  }
  MSE_test_loss <- tf$reduce_mean(((1-M) * X - (1-M)*G_sample)^2)/tf$reduce_mean(1-M)
  
  #Solver####
  D_solver <- tf$train$AdamOptimizer()$minimize(D_loss, var_list = theta_D)
  G_solver <- tf$train$AdamOptimizer()$minimize(G_loss, var_list = theta_G)
  
  sess = tf$Session()
  sess$run(tf$global_variables_initializer())
  
  #Start Training#####
  
  for(it in (1:iterations)){
    trn_idx <-  sample(1:length(idx), mb_size)
    X_mb <- trainX[trn_idx,] %>% as.matrix()
    C_mb <- trainC[trn_idx,] %>% as.matrix()
    Z_mb <- sample.Z(mb_size, Dim) %>% as.matrix()
    M_mb <- trainM[trn_idx,] %>% as.matrix()
    H_mb <- sample.M(mb_size, Dim, 1-p_hint) %>% as.matrix()
    H_mb <- M_mb * H_mb %>% as.matrix()
    
    New_X_mb <- M_mb*X_mb + (1-M_mb) * Z_mb
    
    D_loss_curr <- sess$run(c(D_solver, D_loss1), feed_dict = dict(M=M_mb, New_X = New_X_mb, H=H_mb, Covar_X = C_mb))
    G_loss_curr <- sess$run(c(G_solver, G_loss, MSE_train_loss, MSE_test_loss, G_loss1), 
                            feed_dict = dict(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb, Covar_X = C_mb))
    
    if(it %% 100==0) {
      print(paste0('Iter:', it))
      print(paste0('Train_loss: ', G_loss_curr[3] %>% as.numeric %>% sqrt() %>% round(4)))
      print(paste0('Test_loss: ',  G_loss_curr[4] %>% as.numeric %>% sqrt() %>% round(4)))
      # print(paste0('Expend Equality: ',  D_loss_curr[3] %>% as.numeric %>% sqrt() %>% round(4)))
      # print(paste0('Diff Charge: ',  D_loss_curr[4] %>% as.numeric %>% sqrt() %>% round(4)))
      print(paste0('GLoss: ',  G_loss_curr[2] %>% as.numeric %>% sqrt() %>% round(4)))
      print(paste0('GLoss: ',  G_loss_curr[5] %>% as.numeric %>% sqrt() %>% round(4)))
      
      print(paste0('DLoss: ',  D_loss_curr[2] %>% as.numeric %>% sqrt() %>% round(4)))
      
    }
  }
  
  X_mb <- testX
  Z_mb <- sample.Z(testX %>% nrow(), Dim)
  M_mb <- testM
  C_mb <- testC
  
  New_X_mb <- M_mb*X_mb + (1-M_mb) * Z_mb
  
  Test <- sess$run(c(MSE_test_loss, G_sample), feed_dict = dict(X=testX, M=testM, New_X=New_X_mb, Covar_X = C_mb))
  paste0('Final Test RMSE: ', Test[1] %>% as.numeric %>% sqrt %>% round(4)) %>% print
  
  
#output final fully imputed data set 
  X_mb <- data
  Z_mb <- sample.Z(X_mb %>% nrow(), Dim)
  M_mb <- Missing
  C_mb <- covariates
  
  New_X_mb <- M_mb*X_mb + (1-M_mb) * Z_mb
  
  Final <- sess$run(c(MSE_test_loss, G_sample), feed_dict = dict(X=X_mb, M=M_mb, New_X=New_X_mb, Covar_X = C_mb))
  
  final_imput <- M_mb*X_mb + (1-M_mb)*Final[[2]]
  #final_imput <- final_imput %>% select(names(data))
  
  # #inflate to original values.
  # for (i in 1:length(inflation_factors)){
  #   Final[[2]][,i] <- Final[[2]][,i]*10^inflation_factors[[i]]
  #   final_imput[,i] <- final_imput[,i]*10^inflation_factors[[i]]
  # }
  # 
  
  return(list(imputed = final_imput, sample = Final[[2]], actual = X_mb, missing.mask = M_mb))
}