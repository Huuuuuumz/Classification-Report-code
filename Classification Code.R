---
title: "Classification Report Code"
student number: qwlf81
output: html_document
date: '2022-03-20'
---
 Read Data
```{r}
fatal_mi <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/heart_failure.csv")
View(fatal_mi)
```

 Data Summary
```{r}
library("skimr")
skim(fatal_mi)
```

 Data Visualisations
```{r}
DataExplorer::plot_histogram(fatal_mi, ncol = 3)
```


```{r}
DataExplorer::plot_boxplot(fatal_mi, by = "fatal_mi", ncol = 3)
```

Model Fitting

```{r}
library("tidyverse")
library("ggplot2")
fatal_mi <- fatal_mi %>%
  select(-sex, -time)
```


```{r}
library("mlr3")
fatal_mi$fatal_mi<-as.factor(fatal_mi$fatal_mi)
task_fatalmi <- TaskClassif$new(id = "FATAL",
                               backend = na.omit(fatal_mi),
                               target = "fatal_mi")
task_fatalmi
```

```{r}
library("tidyverse")
library("ggplot2")
library(caret)
set.seed(200)
train<-createDataPartition(y=fatal_mi$fatal_mi,p=0.80,list=FALSE)
trainset<-fatal_mi[train,]
testset<-fatal_mi[-train,]

trainset$fatal_mi<-as.factor(trainset$fatal_mi)
trainset_fatalmi <- TaskClassif$new(id = "FATAL",
                               backend = na.omit(trainset),
                               target = "fatal_mi")
trainset_fatalmi

testset$fatal_mi<-as.factor(testset$fatal_mi)
testset_fatalmi <- TaskClassif$new(id = "FATAL",
                               backend = na.omit(testset),
                               target = "fatal_mi")
testset_fatalmi
```

Define Learners:
logistic regression learner
SVM learner
Classification trees learner
```{r}
library("mlr3learners")
library("mlr3proba")
learner1 <- lrn("classif.log_reg")
learner1
learner2 <- lrn("classif.svm")
learner2
learner3 <- lrn("classif.rpart")
learner3
```


```{r}
learner1$predict_type = "prob"
learner1$train(trainset_fatalmi)
pred1 <- learner1$predict(testset_fatalmi)
pred1
```


```{r}
pred1$score(msr("classif.acc"))
pred1$confusion
```


```{r}
learner2$predict_type = "prob"
learner2$train(trainset_fatalmi)
pred2 <- learner2$predict(testset_fatalmi)
pred2
```


```{r}
pred2$score(msr("classif.acc"))
pred2$confusion
```


```{r}
learner3$predict_type = "prob"
learner3$train(trainset_fatalmi)
pred3 <- learner3$predict(testset_fatalmi)
pred3
```


```{r}
pred3$score(msr("classif.acc"))
pred3$confusion
```

Data resampling
```{r}
library("data.table")
library("mlr3verse")
```
Cross Validation
```{r}
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(task_fatalmi)
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_svm <- lrn("classif.svm", predict_type = "prob")
lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
```
```{r}
res_lr <- resample(task_fatalmi, lrn_lr, cv5, store_models = TRUE)
res_svm <- resample(task_fatalmi, lrn_svm, cv5, store_models = TRUE)
res_rpart <- resample(task_fatalmi, lrn_rpart, cv5, store_models = TRUE)
# Look at accuracy
res_lr$aggregate()
res_svm$aggregate()
res_rpart$aggregate()
```
```{r}
res <- benchmark(data.table(
  task       = list(task_fatalmi),
  learner    = list(lrn_lr,lrn_svm,
                    lrn_rpart),
  resampling = list(cv5)
), store_models = TRUE)
res
res$aggregate()
```
Bootstrap
```{r}
BS<- rsmp("bootstrap")
BS$instantiate(task_fatalmi)
lrn_lr1 <- lrn("classif.log_reg", predict_type = "prob")
lrn_svm1 <- lrn("classif.svm", predict_type = "prob")
lrn_rpart1 <- lrn("classif.rpart", predict_type = "prob")
```
```{r}
res_lr1 <- resample(task_fatalmi, lrn_lr1, BS, store_models = TRUE)
res_svm1 <- resample(task_fatalmi, lrn_svm1, BS, store_models = TRUE)
res_rpart1 <- resample(task_fatalmi, lrn_rpart1, BS, store_models = TRUE)
# Look at accuracy
res_lr1$aggregate()
res_svm1$aggregate()
res_rpart1$aggregate()
```
```{r}
res1 <- benchmark(data.table(
  task       = list(task_fatalmi),
  learner    = list(lrn_lr1,lrn_svm1,
                    lrn_rpart1),
  resampling = list(BS)
), store_models = TRUE)
res1
res1$aggregate()
```



```{r}
autoplot(res)
autoplot(res1)
```
```{r}
lrn_cart_cp <- lrn("classif.log_reg", predict_type = "prob")

res <- benchmark(data.table(
  task       = list(task_fatalmi),
  learner    = list(lrn_lr1,lrn_svm1,
                    lrn_rpart1,
                    lrn_cart_cp),
  resampling = list(BS)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
```
```{r}
trees <- res$resample_result(3)
res$resample_result(3)
tree1 <- trees$learners[[1]]
tree1_rpart <- tree1$model
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)
```
```{r}
plot(res$resample_result(3)$learners[[5]]$model, compress = TRUE, margin = 0.1)
text(res$resample_result(3)$learners[[5]]$model, use.n = TRUE, cex = 0.8)
```
```{r}
library(ggplot2)
autoplot(res1$resample_result(1), type = "roc")
```




