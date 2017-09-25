Course Project
================

Background
----------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

Data
----

The training data for this project are available here: \[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>\]

The test data are available here: \[<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>\]

The data for this project come from this source: \[<http://groupware.les.inf.puc-rio.br/har>\]. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

The classe variable contains 5 different ways barbell lifts were performed correctly and incorrectly:

-   Class A: exactly according to the specification

-   Class B: throwing the elbows to the front

-   Class C: lifting the dumbbell only halfway

-   Class D: lowering the dumbbell only halfway

-   Class E: throwing the hips to the front

Objective
---------

The goal of this project is to predict the manner in which people performed barbell lifts. This is the classe variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Loading the data
----------------

Packages used for analysis. This assumes the packages are already installed. Use the install.packages("") command if a package not installed yet.

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(rpart)
library(rpart.plot)
library(RColorBrewer)
```

Load the data into R

``` r
# The location where the training data is to be downloaded from
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# The location where the testing data is to be downloaded from
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Reading/loading the training data 
train_data <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
# Reading/loading the testing data in your working directory
test_data <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

# Take a look at the Training data classe variable
summary(train_data$classe)
```

    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

Partitioning the data for Cross-validation
------------------------------------------

The training data is split into two data sets, one for training the model and one for testing the performance of our model. The data is partitioned by the classe variable, which is the varible we will be predicting. The data is split into 60% for training and 40% for testing.

``` r
inTrain <- createDataPartition(y=train_data$classe, p = 0.60, list=FALSE)
training <- train_data[inTrain,]
testing <- train_data[-inTrain,]

dim(training)
```

    ## [1] 11776   160

``` r
dim(testing)
```

    ## [1] 7846  160

Data Processing
---------------

Drop the first 7 variables because these are made up of metadata that would cause the model to perform poorly.

``` r
training <- training[,-c(1:7)]
```

Remove NearZeroVariance variables

``` r
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]
```

There are a lot of variables where most of the values are 'NA'. Drop variables that have 60% or more of the values as 'NA'.

``` r
training_clean <- training
for(i in 1:length(training)) {
  if( sum( is.na( training[, i] ) ) /nrow(training) >= .6) {
    for(j in 1:length(training_clean)) {
      if( length( grep(names(training[i]), names(training_clean)[j]) ) == 1)  {
        training_clean <- training_clean[ , -j]
      }   
    } 
  }
}

# Set the new cleaned up dataset back to the old dataset name
training <- training_clean
```

Transform the test\_data dataset

``` r
# Get the column names in the training dataset
columns <- colnames(training)
# Drop the class variable
columns2 <- colnames(training[, -53])
# Subset the test data on the variables that are in the training data set
test_data <- test_data[columns2]
dim(test_data)
```

    ## [1] 20 52

Cross-Validation: Prediction with Random Forest
-----------------------------------------------

A Random Forest model is built on the training set. Then the results are evaluated on the test set

``` r
set.seed(54321)
modFit <- randomForest(classe ~ ., data=training)
prediction <- predict(modFit, testing)
cm <- confusionMatrix(prediction, testing$classe)
print(cm)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2229    5    0    0    0
    ##          B    2 1505   16    0    0
    ##          C    1    8 1347   13    2
    ##          D    0    0    5 1272    3
    ##          E    0    0    0    1 1437
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9929          
    ##                  95% CI : (0.9907, 0.9946)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.991           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9987   0.9914   0.9846   0.9891   0.9965
    ## Specificity            0.9991   0.9972   0.9963   0.9988   0.9998
    ## Pos Pred Value         0.9978   0.9882   0.9825   0.9938   0.9993
    ## Neg Pred Value         0.9995   0.9979   0.9968   0.9979   0.9992
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2841   0.1918   0.1717   0.1621   0.1832
    ## Detection Prevalence   0.2847   0.1941   0.1747   0.1631   0.1833
    ## Balanced Accuracy      0.9989   0.9943   0.9905   0.9939   0.9982

``` r
overall.accuracy <- round(cm$overall['Accuracy'] * 100, 2)
sam.err <- round(1 - cm$overall['Accuracy'],2)
```

The model is 99.29% accurate on the testing data partitioned from the training data. The expected out of sample error is roughly 0.01.

``` r
plot(modFit)
```

![](unnamed-chunk-9-1.png)

In the above figure, error rates of the model are plotted over 500 trees. The error rate is less than 0.04 for all 5 classe.

Cross-Validation: Prediction with a Decision Tree
-------------------------------------------------

``` r
set.seed(54321)
modFit2 <- rpart(classe ~ ., data=training, method="class")
prediction2 <- predict(modFit2, testing, type="class")
cm2 <- confusionMatrix(prediction2, testing$classe)
print(cm2)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1974  244   36   66   48
    ##          B   65  942  120  109  113
    ##          C   52  144 1093  197  170
    ##          D   72  122   90  828   69
    ##          E   69   66   29   86 1042
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7493          
    ##                  95% CI : (0.7396, 0.7589)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6823          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8844   0.6206   0.7990   0.6439   0.7226
    ## Specificity            0.9298   0.9357   0.9131   0.9462   0.9610
    ## Pos Pred Value         0.8336   0.6983   0.6600   0.7011   0.8065
    ## Neg Pred Value         0.9529   0.9113   0.9556   0.9313   0.9390
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2516   0.1201   0.1393   0.1055   0.1328
    ## Detection Prevalence   0.3018   0.1719   0.2111   0.1505   0.1647
    ## Balanced Accuracy      0.9071   0.7781   0.8560   0.7950   0.8418

``` r
overall.accuracy2 <- round(cm2$overall['Accuracy'] * 100, 2)
sam.err2 <- round(1 - cm2$overall['Accuracy'],2)
```

The model is 74.93% accurate on the testing data partitioned from the training data. The expected out of sample error is roughly 0.25.

Plot the decision tree model

``` r
rpart.plot(modFit2)
```

![](unnamed-chunk-11-1.png)

Prediction on the Test Data
---------------------------

The Random Forest model gave an accuracy of 99.29, which is much higher than the 74.93% accuracy from the Decision Tree. So we will use the Random Forest model to make the predictions on the test data to predict the way 20 participates performed the exercise.

``` r
final_prediction <- predict(modFit, test_data, type="class")
print(final_prediction)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

Conclusions
-----------

There are many different machine learning algorithms. I chose to compare a Random Forest and Decision Tree model. For this data, the Random Forest proved to be a more accurate way to predict the manner in which the exercise was done.
