# MIS-510-Portfolio-Project-1.1

This is my second repository on GitHub.  I hope you like it as I was very proud of it when I first put it together.

---
title: "MIS 510 Portfolio Project Option 1"
author: "John Midkiff"
date: "7/2/2020"
output: word_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown
#Importing GermanCredit dataset
```{r}
library(readr)
german.df <- read_csv("GermanCredit.csv")
```

##Investigating the distribution and structure of the GermanCredit dataset using the summary() function to identify median, mean, minimum and maximum values, and the first and third quartiles.  This helps the investigator understand the distribution of the data such as spread and if the median is close to the mean.
```{r}
summary(german.df)
```

#Similar to the summary function, the describe function helps the investigator understand what the data looks like.  In addition to providing the mean it shows the quality of the data by providing the number of missing values for each variable but how many distinct values are in each column.  The results show that many of the columns have only 2 values and may make good candidates for a logistic regression analysis.  The results also show that there is no missing values in any of the columns.
```{r}
library(Hmisc)
describe(german.df)
```

#Variables that are highly correlated with each other can negatively affect the results of regression analysis.  Therefore, it is good to understand the relationship the different variables have so that highly correlated variables can be removed before model generation.  As only 3 of the 32 columns are continuous these three will be analyzed for correlation.  The results of the scatterplot matrix and the correlation matrix show that these three variables are not highly correlated enough to warrant their removal from the dataset.
```{r}
library(GGally)
ggpairs(german.df[, c(3, 11, 23)])
library(corrplot)
source("http://www.sthda.com/upload/rquery_cormat.r")
rquery.cormat(german.df[, c(3,11,23)])
```

#Boxplots are a good way to reveal patterns between a continuous and a categorical variable.  In the GermanCredit dataset the variable Response, corresponding to approval, is categorical, being either a no (0) or a yes (1).  To explore the relationship between the response and the continuous variables boxplots were generated using the boxplot function.
```{r}
boxplot(german.df$DURATION ~ german.df$RESPONSE, xlab = "Response", ylab = "Duration of Checking Account", notch = TRUE, col = c("green", "purple"))
```

#Histograms show how the data is distributed by separating the range of values into discrete packets called bins and then adding up the number of values that would fall in each bin.  This can show whether a distribution is normally distributed, left skewed, or right skewed.  From the graph, the distribution of the ages of credit applicants is left skewed.  The added curve shows the mean and the standard deviation of the ranges of age. 
```{r}
hist(german.df$AGE, freq = FALSE, xlab = "Age of Credit Applicant", main = "Distribution of Ages", col = "pink")
curve(dnorm(x, mean = mean(german.df$AGE), sd = sd(german.df$AGE)), add = TRUE, col = "darkblue", lwd = 2)
```

##Dividing the dataset into a training and validation datasets
#Partitioning the data is an essential step in supervised learning models.  Separating the data into a training and validation dataset addresses the problem of overfitting the model.  In a regression model, the training set is used to create the model and it is tested against the validation dataset and the prediction errors of expected versus actual values is determined.  This helps determine how well the model describes the data.  In a classification model the number of misclassified records and accurately classified items are used in calculations to determine accuracy, precision, and other factors that evaluate the model
```{r}
train.index <- sample(c(1:1000), 600)
train.df <- german.df[train.index, ]
valid.df <- german.df[-train.index, ]
```

##Generating the logistic regression model
#The model will be generated using with the family set as binomial for a logistic regression model.  The column OBs will not be included as it is the row number and is not actual data.
```{r}
logit.reg <- glm(RESPONSE ~. -`OBS#`, data = train.df, family = "binomial")
options(scipen=999)
```

#Using the summary function the independent variables that have a significant influence upon the dependent variable RESPONSE are shown.  Those variables that have the most statistically significant impact on whether a loan request is yes or no are: the ownership of a checking and savings account, banking history, and Installment rate as a percentage of disposable income. 
```{r}
summary(logit.reg)
```

##Evaluating the model using the validation dataset.
#Having created the model using the training dataset the next step is to evaluate the model using the validation dataset.  First, the predictions will be created for all of the different records.  
```{r}
library(gains)
logit.reg.pred <- predict(logit.reg, valid.df, type = "response")
```

#A lift chart shows how the model performs against a model of random selection as the benchmark.  The results of the plotting benchmark and how the model performs with the validation dataset shows that the model is a modest improvement over the benchmark model.  At its best with 50% of the validation dataset evaluated by the model the lift is approximately 30%.
```{r}
gain <- gains(valid.df$RESPONSE, logit.reg.pred, groups = length(logit.reg.pred))
data.frame(actual = valid.df$RESPONSE[1:5], predicted = logit.reg.pred[1:5])
plot(c(0, gain$cume.pct.of.total * sum(valid.df$RESPONSE)) ~ c(0, gain$cume.obs), xlab = "Number of Applicants", ylab = "Cumulative", main = "", type = "l")
lines(c(0, sum(valid.df$RESPONSE)) ~ c(0, dim(valid.df)[1]), lty = 2)
```

#A confusion matrix is a standard method for evaluating a logistic regression or classification model.  A confusion matrix shows the number of true positives and true negatives along with the number of false postives and false negatives.  The accuracy is the number of correctly classified items, whether positive or negative.  The sensitivity is the number of true positives correctly identified divided by all positives values in the dataset for the variable.  Specificity is the percentage of true negatives identified from the total number of true negatives in the dataset.  The results from the confusion matrix of the validation dataset show that the accuracy is quite low at almost 73% as is the sensitivity, which is at 45%.  The specificity is much higher at 86% which in the case of loan approval it may be more important to identify those who would be denied a loan rather than lose someone who would be approved in a risk-cost analysis.
```{r}
library(caret)
reg.pred <- predict(logit.reg, valid.df, type = "response")
reg.pred <- ifelse(reg.pred > 0.5, 1, 0)
reg.pred <- factor(reg.pred)
resp.df1 <- factor(valid.df$RESPONSE)
confusionMatrix(reg.pred, resp.df1)
lvs1 <- c("actual", "predicted")
truth1 <- factor(rep(lvs1, times = c(281, 119)), levels = (lvs1))
pred1 <- factor(c(rep(lvs1, times = c(241, 40)), rep(lvs1, times = c(57, 62))), levels = rev(lvs1))
xtab <- table(pred1, truth1)
cm <- confusionMatrix(pred1, truth1)
cm$table
fourfoldplot(cm$table)
```

##Classification Tree Analysis
#Classification trees are a method of classification that enables the investigator to see visually the process that a model takes in determining into which class each record belongs in.  It does this by separating the independent variables into subgroups and creates splits from these predictors.  The reasoning behind the model can be paraphrased into "if-then" statements, as in "If this variable has this value then the classification is this."  The terminal nodes show the results of all of the previous decision nodes and provide the models rationale for selecting the classification it has.  This initial classification tree has only 2 nodes but, as more nodes are added, the accuracy of the model increases.  
```{r}
library(rpart)
library(rpart.plot)
class.tree <- rpart(RESPONSE ~ ., data = train.df, control = rpart.control(maxdepth = 2), method = "class")
prp(class.tree, type = 1, extra = 1, split.font = 1, varlen = -10)
```

#This is a classification tree of 8 nodes.  As more nodes have been added the number of records in the terminal trees gets smaller accuracy increases.  In this classification tree instead of the quantities of number of applicants that would be classified with either good or bad credit the percentage is given.  For example, the far left terminal leaf, if an applicant has a credit history value below 2 and a savings account value below 2 there is 10% chance they will be considered a good credit risk.
```{r}
fit <- rpart(RESPONSE ~., data = train.df, control = rpart.control(maxdepth = 4), method = "class")
rpart.plot(fit, extra = 106)
```


#Default classification tree with training dataset
```{r}
default.ct <- rpart(RESPONSE ~ ., data = train.df, method = "class")
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)
```

#Creating a full classification tree for all of the data in the training set.  At this level all variables become either a decision or terminal node.  
```{r}
deeper.ct <- rpart(RESPONSE ~ ., data = train.df, method = "class", cp = 0, minsplit = 1)
length(deeper.ct$frame$var[deeper.ct$frame$var == "<leaf>"])
prp(deeper.ct, type = 1, under = TRUE, split.font = 1, varlen = -10, box.col = ifelse(deeper.ct$frame$var == "<leaf>", 'gray', 'white'))
```

#The confusion matrix for the default classification tree shows that the accuracy of this model has exceeded that of the multiple logistic regression, as has specificity and sensitivity.  
```{r}
default.ct.point.pred.train <- predict(default.ct, train.df, type = "class")
class.df1 <- factor(train.df$RESPONSE)
str(class.df1)
confusionMatrix(default.ct.point.pred.train, class.df1)
```

#When the validation dataset it tested on the model created by the training results the performance is slightly less than that of the training set.  While specificity, the true negative rate, remains above 90% both accuracy and sensitivity have dropped markedly.  As a real world scenario honing the model by trimming leaves to reduce the number of decision points but retaining those that have the most effect on a correct classification could improve the results.
```{r}
default.ct.point.pred.valid <- predict(default.ct, valid.df, type = "class")
class.df2 <- factor(valid.df$RESPONSE)
confusionMatrix(default.ct.point.pred.valid, class.df2)
```

#When the classification tree is expanded to include all possible variables the accuracy, sensitivity, and specificity are all at 100%.  While the model is perfect at explaining the training dataset this is a case of overtraining where modeling the training dataset to provide perfect results will show a decreased ability to classify correctly other datasets.
```{r}
deeper.ct.point.pred.train <- predict(deeper.ct, train.df, type = "class")
confusionMatrix(deeper.ct.point.pred.train, class.df1)
```

#When the validation dataset is entered into the deeper model from the training dataset the results are significantly worse than the other models.  Accuracy is 67% and specificity and sensitivity are 42% and 79%, respectively.  This example perfectly demonstrates the dangers of overfitting when modeling without splitting into a training and validation dataset or focusing only upon the performance of the training dataset.
```{r}
deeper.ct.point.pred.valid <- predict(deeper.ct, valid.df, type = "class")
confusionMatrix(deeper.ct.point.pred.valid, class.df2)
```
