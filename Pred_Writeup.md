---
title: "Prediction Assignment Writeup"
author: "David Knaack"
date: "November 17, 2019"
output: html_document
---

## Required Packages

```{r, results='hide'}
library(ggplot2)
library(caret)
library(fscaret)
library(randomForest)
library(e1071)
```

## Getting the Data
I first get our data with the given links.
```{r}
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", stringsAsFactors=FALSE)
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", stringsAsFactors=FALSE)
training$classe = as.factor(training$classe)
```

## Splitting the data and selecting features
I set the seed to make this analysis reproducible and we need to split the original training set into our training set and a validation set.
```{r}
set.seed(333)
inTrain <- createDataPartition(y=training$classe, p=0.7, list=F)
training1 <- training[inTrain, ]
training2 <- training[-inTrain, ]
```

Many columns of the data set contain the same value accros the lines. These “near-zero variance predictors”" bring almost no information to our model and will make computing unnecessarily longer. Others are entirely filled with NA values. Finnaly, the six first variables do not concern fitness motions whatsoever. They also need to be remove before we start fitting our model.
```{r}
#removing near-zero variance predictors
nzv <- nearZeroVar(training)
training1 <- training1[, -nzv]
training2 <- training2[, -nzv]
#removing predictors with NA values
training1 <- training1[, colSums(is.na(training1)) == 0]
training2 <- training2[, colSums(is.na(training2)) == 0]
#removing columns unfit for prediction (ID, user_name, raw_timestamp_part_1 etc ...)
training1 <- training1[, -(1:5)]
training2 <- training2[, -(1:5)]
```

## Selecting a model
We chose to fit a random forest model. This model provided the most accurate results all along the machine learning course.The cross-validation is set to draw a subset of the data three different times.
```{r}
mod1 <- train(as.factor(classe) ~., method = "rf", data = training1, verbose = TRUE, trControl = trainControl(method="cv"), number = 3)
pred1 <- predict(mod1, training1)
confusionMatrix(pred1, training1$classe)
```

We get a very high accuracy of 99% but we still need to know how this model performs against the test set before expressing a conclusion.
```{r}
pred2 <- predict(mod1, training2)
confusionMatrix(pred2, training2$classe)
```
As we can see we still get a very high accuracy. We didn’t overfit the model when training it.

## Testing the model
We apply to the final test set the same features selection method that we use for the training set.
```{r}
testing <- testing[, colSums(is.na(testing)) == 0]
testing <- testing[, -(1:5)]
nzvt <- nearZeroVar(testing)
testing <- testing[, -nzvt]
```

We test the random forest model on the test set.
```{r}
pred3 <- predict(mod1, testing)
pred3
```