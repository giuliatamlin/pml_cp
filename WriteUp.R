library(caret)
library(ggplot2)
library(RColorBrewer)

#Dataset from
#Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, 
#H. Qualitative Activity Recognition of Weight Lifting Exercises. 
#Proceedings of 4th International Conference in Cooperation with 
#SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
#Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3uJG8yOEC


train_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv" 

directory = "."
setwd(directory)
trainfile = "train.csv"
testfile = "test.csv"
train_dest = paste(directory, trainfile,sep ="/")
test_dest = paste(directory, testfile,sep ="/")


# if (!file.exists(directory)){
#   dir.create(directory)
# }

if (!file.exists(train_dest) | !file.exists(test_dest)){
  method = "curl"  
  download.file(train_url,destfile = train_dest,method = method)
  download.file(test_url,destfile = test_dest,method = method)
}

train_raw <- read.csv(train_dest)
test_raw <-read.csv(test_dest)

#first step: clean and explore training set

#remove columns with missing values

cols = apply(train_raw,2,function(x) {sum(is.na(x))==0})
train_raw = train_raw[,cols]
#remove the same columns from  the testing dataset
test_raw = test_raw[,cols]

#remove columns containing "#DIV/0!" as it is not a meaningful value

cols = apply(train_raw,2,function(x) {length(grep("#D",x))==0} )
train_raw = train_raw[,cols]
#remove same columns from the testing dataset 
test_raw = test_raw[,cols]
#number of columns in the datasets reduced to 60


#remove other columns which are not useful for the prediction i.e. those
#containing user names, timestamps and time windows
train_raw = train_raw[,-grep("user_name|window|timestamp|^X",names(train_raw))]
test_raw = test_raw[,-grep("user_name|window|timestamp|^X",names(test_raw))]
# number of columns in the datasets reduced to 53

# now that the most obvious cleaning has been performed let us see if any of the 
#remaining variables are highly correlated 

#construct correlation matrix
corrmat = abs(cor(train_raw[,-53]))
#set diag to 0
diag(corrmat) = 0

#if any pairs of variables is highly correlated then
# the dimensionality of the set can be further reduced
if (length(which(corrmat>0.8,arr.ind = T)) == 0){
  train = train_raw
  test = test_raw
} else{# perform PCA analysis
  
  preobj = preProcess(train_raw[,-53],method = "pca", thresh = 0.95)
  trainPC = predict(preobj,train_raw[,-53])
  testPC = predict(preobj,test_raw[,-53])
  # number of columns reduced to 25
  train = trainPC
  test = testPC
  
}

#now that basic preprocessing has been done we proceed to data slicing
#and set the seed for reproducibility

set.seed(20140604)

# split the train dataset in two parts for validation
cvind = createDataPartition(train_raw$classe,p = 0.7,list = FALSE)

train_cv = train[cvind,]
#pick up the class variable values from the test_raw set against which 
#train the model
classe_train_cv = train_raw$classe[cvind]

test_cv = train[-cvind,]
#pick up the class variable values from the test_raw set against which 
#cross-validate the model
classe_test_cv = train_raw$classe[-cvind]


#model training: I choose to use a decision tree as this class of algorithms
#seems to me more  suited for multi-variate classification problems.
# I will compare decision tree vs random forests and pick one of the two


#choose the parameters for cross validation

controls1 = trainControl(method = "cv", number = 5)
# in order to optimize performance I set the parameter number of subsets
# to the default i.e.
# sqrt(p) with p = ncol, in this case number = 5
#setting number = 10, it would be interesting to compare the cv against repeated
#cv i.e.
#controls2 = trainControl(method = "repeatedcv", 
#number =2 , repeats = 5 )

#in order to improve a little bit computation speed I decided to set mtry to 18.
# The reason for the choice of this number is that, after PCA, all variables, being
# linearly independent, are likely to contribute on a split. On the other hand
# I wanted to prevent overfitting and gain some computational speed, so
#  trying  approx 2/3 of the available variables seemed a good compromise.

if (!exists("modFitDecTree")){
  modFitDecTree <- train(classe_train_cv ~.,
                         data = train_cv,
                         method="rpart2",
                         trControl = controls1,
                         tuneGrid = data.frame(.maxdepth = 30))
  
  
}
print(modFitDecTree$finalModel)

#random forest
if (!exists("modFit1")){
  modFit1 <- train(classe_train_cv ~.,
                   data = train_cv,
                   method="rf",
                   trControl = controls1,
                   prox = TRUE,
                   allowParallel = TRUE, ntrees  = 250, 
                   tuneGrid = data.frame(.mtry = 18))
}

print(modFit1$finalModel)
# Random Forest 
# 
# 13737 samples
# 24 predictor
# 5 classes: 'A', 'B', 'C', 'D', 'E' 
# 
# No pre-processing
# Resampling: Cross-Validated (5 fold) 
# Summary of sample sizes: 10991, 10989, 10990, 10988, 10990 
# Resampling results
# 
# Accuracy  Kappa      Accuracy SD  Kappa SD   
# 0.960253  0.9497201  0.002709292  0.003433906
# 
# Tuning parameter 'mtry' was held constant at a value of 18

# Let us first check how the fitted model does against the train set
predict0dt = predict(modFitDecTree,train_cv)
confusionMatrix(classe_train_cv,predict0dt)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 3371  108  131  228   68
# B 1208  526  186  306  432
# C 1575  102  372   84  263
# D  706   65  227  864  390
# E  752  324   69  201 1179
# 
# Overall Statistics
# 
# Accuracy : 0.4595          
# 95% CI : (0.4511, 0.4679)
# No Information Rate : 0.5541          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2914          
# Mcnemar's Test P-Value : <2e-16          
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.4429  0.46756  0.37766   0.5134  0.50557
# Specificity            0.9127  0.83095  0.84128   0.8849  0.88198
# Pos Pred Value         0.8630  0.19789  0.15526   0.3837  0.46693
# Neg Pred Value         0.5686  0.94593  0.94595   0.9287  0.89716
# Prevalence             0.5541  0.08190  0.07170   0.1225  0.16976
# Detection Rate         0.2454  0.03829  0.02708   0.0629  0.08583
# Detection Prevalence   0.2843  0.19349  0.17442   0.1639  0.18381
# Balanced Accuracy      0.6778  0.64926  0.60947   0.6991  0.69378



predict0rf = predict(modFit1,train_cv)
confusionMatrix(classe_train_cv,predict0rf)


# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 3906    0    0    0    0
# B    0 2658    0    0    0
# C    0    0 2396    0    0
# D    0    0    0 2252    0
# E    0    0    0    0 2525
# 
# Overall Statistics
# 
# Accuracy : 1          
# 95% CI : (0.9997, 1)
# No Information Rate : 0.2843     
# P-Value [Acc > NIR] : < 2.2e-16  
# 
# Kappa : 1          
# Mcnemar's Test P-Value : NA         
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
# Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
# Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
# Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
# Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
# Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
# Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
# Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

#Can the fact that the fitted model has accuracy could be a warning sign
#of overfitting?

predict1 = predict(modFit1,test_cv)

confusionMatrix(classe_test_cv,predict1)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    A    B    C    D    E
# A 1656    7    5    6    0
# B   12 1115    5    0    7
# C    4   25  977   16    4
# D    5    2   42  910    5
# E    1    4    6    2 1069
# 
# Overall Statistics
# 
# Accuracy : 0.9732          
# 95% CI : (0.9687, 0.9771)
# No Information Rate : 0.2851          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.966           
# Mcnemar's Test P-Value : 0.0003989       
# 
# Statistics by Class:
# 
#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            0.9869   0.9670   0.9440   0.9743   0.9853
# Specificity            0.9957   0.9949   0.9899   0.9891   0.9973
# Pos Pred Value         0.9892   0.9789   0.9522   0.9440   0.9880
# Neg Pred Value         0.9948   0.9920   0.9881   0.9951   0.9967
# Prevalence             0.2851   0.1959   0.1759   0.1587   0.1844
# Detection Rate         0.2814   0.1895   0.1660   0.1546   0.1816
# Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
# Balanced Accuracy      0.9913   0.9810   0.9669   0.9817   0.9913


#Now try the model on the test dataset
predict_on_test = predict(modFit1,test)

#write the prediction on separate files

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

answers = as.character(predict_on_test)

answers_dest = paste(directory, "prediction",sep ="/")
if (!file.exists(answers_dest)){
  dir.create(answers_dest)
}

setwd(answers_dest)

pml_write_files(answers)