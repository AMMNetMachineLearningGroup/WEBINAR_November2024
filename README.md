---
title: "Building and Interpreting AI/ML Models for Malaria Prediction in R"
author: "D.K.Muriithi"
date: November 2024
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
	echo = TRUE,
	fig.height = 4,
	fig.width = 8,
	message = FALSE,
	warning = FALSE,
	comment = NA)
```

## Introduction
We will explore building interpretable AI/ML models for malaria prediction and analysis in R, focusing on   libraries like caret, iml, lime, and DALEX.

## Overview
- **Objective**: Develop interpretable machine learning models for malaria risk prediction
- **Significance**: Improving early detection and intervention strategies
- **Approach**: Utilizing R for transparent and explainable predictive modeling

## Malaria: A Global Health Challenge (WHO 2022 report)
- Affects over 249 millions annually, particularly in developing regions (Africa is home to 94% of cases)
- Reported death 608,000 with 95% in Africa(580,000)
- Complex interactions between:
  - Patient-level factors
  - Environmental factors
  - Parasite information

## Data Sources and Preprocessing
## Key Data Components
1. **Patient-levelData**
   - Age
   - Gender
   - Symptoms (e.g., fever, chills)
   - Malaria test results
   - Medical history

2. **Environmental Predictors**
   - Temperature
   - Rainfall
   - Endemic zones
   - Humidity
   
3. **Parasite information**
   - Type of malaria parasite(Plasmodium falciparum or Plasmodium vivax...))
   - Mosquito Density
   
4. **Target Variable**
   - Malaria test result (binary: positive/negative)
  
## Importance of Interpretability in Malaria Prediction
1. **Trust and Transparency**
   - Stakeholders (e.g., healthcare professionals and policymakers) are more likely to adopt models they understand.
   
2. **Model Validation**
   - Interpretability ensures that predictions align with domain knowledge, preventing spurious correlations.

3. **Actionable Insights**
   - Helps identify key factors driving malaria spread and outcomes, such as climatic conditions or patient demographics.

4. **Regulatory Compliance**
   - Interpretability can meet requirements for explainability in healthcare systems.
   
###### PROCEDURE ###### 

1. **Preparing the Malaria Dataset**
   - Before training models, ensure your data is clean and well-prepared. 

2. **Training Machine Learning Models**
    - Use the caret package to train models such as Random Forest or Gradient Boosting.

3. **Model Interpretability Techniques**

  a. **Feature Importance using iml(Interpretable Machine Learning)**
    - Quantifies how much each feature contributes to the model predictions
    - Identify most critical predictors of malaria risk
    - Understand complex interactions between variables
   
  b. **SHAP (Shapley Additive Explanations)**
    - Explain individual predictions with SHAP values.
    - Explains individual predictions using game theory
    
  c. **Local Explanations with LIME(Local Interpretable Model-Agnostic Explanations)**
    - Focuses on explaining individual predictions.
    - Flexible enough to work with almost any supervised learning model
    - Generates plots to illustrate feature contributions
    
  d. **Partial Dependence Plots (PDP)**
    - Understand the relationship between features and predictions.
    - Visualizes the relationship between features and the target prediction
    - Understand non-linear relationships
    
```{r}
setwd("~/webinar")
```

```{r}
# Load required libraries
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Define parameters
n <- 1000  # Number of observations

# Generate synthetic data
synthetic_data <- data.frame(
  # Patient-level predictors
  Age = sample(1:80, n, replace = TRUE),  # Age in years
  Gender = sample(c("Male", "Female"), n, replace = TRUE),
  Symptoms = sample(c("Fever", "Chills", "Headache", "Nausea", "None"), n, replace = TRUE),
  
  # Environmental predictors
  Rainfall = round(runif(n, 50, 300), 1),  # Rainfall in mm
  Temperature = round(runif(n, 20, 35), 1),  # Temperature in degrees Celsius
  Humidity = round(runif(n, 30, 90), 1),  # Humidity percentage
  Endemic_Zone = sample(c("Lake Basin", "coastal", "Highland"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2)),
  Mosquito_Density = round(runif(n, 10, 200), 1),  # Mosquito count per sq. km
  
  # Parasite information
  Parasite_Type = sample(c("Plasmodium falciparum", "Plasmodium vivax", "Other"), 
                         n, replace = TRUE, prob = c(0.7, 0.2, 0.1)),
  
  # Target variable
  Malaria_Result = sample(c("Positive", "Negative"), n, replace = TRUE, 
                          prob = c(0.4, 0.6))  # Based on prevalence
)

# Add noise to ensure variability in data
synthetic_data <- synthetic_data %>%
  mutate(
    Malaria_Result = ifelse(
      (Endemic_Zone == "High" & Temperature > 25 & Humidity > 60 & Symptoms %in% c("Fever", "Chills") & 
       Parasite_Type == "Plasmodium falciparum"),
      "Positive", Malaria_Result
    )
  )
# Save the dataset to a CSV file
#write_csv(data, "KenyaMalariaCases.csv")
```

## Installation and loading of necessary packages/libraries

# Loading libraries
```{r}
library(caret) #for training machine learning models
library(psych) ##for description of  data
library(ggplot2) ##for data visualization
library(caretEnsemble)##enables the creation of ensemble models
library(tidyverse) ##for data manipulation
library(mlbench)  ## For benchmarking ML Models
library(flextable) ## To create and style tables
library(mltools) #for hyperparameter tuning
library(tictoc) #for determining the time taken for a model to run
library(ROSE)  ## for random oversampling
library(smotefamily) ## for smote sampling
library(ROCR) ##For ROC curve
library(pROC) ## For visualizing, smoothing, and comparing ROC curves
library(e1071) ## For statistical modeling and  machine learning tasks
library(class) ## For classification using k-Nearest Neighbors and other methods
library(caTools) ## For splitting data into training and testing sets
library(MASS) ## Provides plotting functions and datasets
library(ISLR) ## for practical applications of statistical learning methods
library(boot) ## Useful for performing bootstrap resampling
library(cvTools) ## Contains functions for cross-validation, bootstrapping, and other resampling methods
library(iml)  ## provide tools to analyze and interpret machine learning models
library(lime) ## powerful tools for interpreting machine learning models
library(DALEX) ## powerful tool for interpreting machine learning models. It provides a unified framework for creating model explanations, making it easier to understand both global and local behavior of machine learning models.
```

## Loading and preparing data/Exporatory of the dataset
```{r}
data <-synthetic_data
#View(data)
#data <- na.omit(data)  # Remove missing values
#dim(data)      ## View the Dimension of the Data
#names(data)     ## View the variable/features/column names
#summary(data)    ## Descriptive Statistics
#describe(data)   ## Descriptive Statistics
data$Malaria_Result <- as.factor(data$Malaria_Result)  # Convert to factor
```

## Visualizing the Dataset
```{r}
# Distribution of malaria test results
ggplot(synthetic_data, aes(x = Malaria_Result, fill = Endemic_Zone)) +
  geom_bar(position = "dodge") +
  labs(title = "Malaria Test Results by Endemic Zone", x = "Malaria Result", y = "Count") +
  theme_minimal()

# Distribution of malaria test results
ggplot(synthetic_data, aes(x = Malaria_Result, fill = Parasite_Type)) +
  geom_bar(position = "dodge") +
  labs(title = "Malaria Test Results by Parasite_Type", x = "Malaria Result", y = "Count") +
  theme_minimal()

# Relationship between temperature and malaria results
ggplot(synthetic_data, aes(x = Temperature, fill = Malaria_Result)) +
  geom_density(alpha = 0.5) +
  labs(title = "Temperature Distribution by Malaria Test Result", x = "Temperature (°C)", y = "Density") +
  theme_minimal()
```

# Plot Target variable using ggplot2 function
```{r}
ggplot(data, aes(x = Malaria_Result, fill=Malaria_Result)) + 
  geom_bar() + 
  labs(x = "Malaria_Result", 
       y = "Respondent",
       tittle = "Malaria Diagnosis Results",
       caption = "Source: Synthetic Malaria Dataset") +
    theme_classic()
```

## DATA PARTITION FOR MACHINE LEARNING
```{r}
set.seed(123)  # Set seed for reproducibility

# Create a partition: 75% for training, 30% for testing
index <- createDataPartition(data$Malaria_Result,p = 0.75, list = FALSE)

# Create training and testing sets
train <- data[index, ]
test <- data[-index, ]

# Get the dimensions of your train and test data
dim(train)
dim(test)
```

## VIEW THE MODELS IN CARET
```{r}
models= getModelInfo()
#names(models)
```

# Cross validation technique
```{r}
control <- trainControl(method ="repeatedcv", number=10, repeats=5, sampling='smote', search='random')## For tuning
```

## Basic Workflow with DALEX
"DALEX" is powerful tool for interpreting machine learning models. It provides a unified framework for creating model explanations, making it easier to understand both global and local behavior of machine learning models.

  **Train a Machine Learning Model: Use any model, such as Random Forest or Gradient Boosting.
  **Wrap the Model with an Explainer: Create an explainer object using explain().
  **Generate Explanations: Use various DALEX functions for global or local interpretability.

# Train a Random Forest model
# Random Forests 
This is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce variance.
# mtry
This parameter controls the number of features randomly chosen as candidates for splitting a node in each tree.

```{R}
# Load libraries
library(caret)
library(DALEX)
set.seed(123)
tuneGrid_rf <- expand.grid(mtry = c(2, 4, 6, 8, 12))
tic()
RFModel <- train(factor(Malaria_Result)~., 
                 data=train, 
                 method="rf", 
                 trControl=control, 
                 tuneGrid=tuneGrid_rf, 
                 na.action = na.omit)
toc()
#RFModel
# Prediction using RF model
RFpred=predict(RFModel,newdata = test)

# Evaluation of the RF model performance metrics
RF_CM<- confusionMatrix(RFpred,as.factor(test$Malaria_Result), positive = "Positive", mode='everything')
M2<- RF_CM$byClass[c(1, 2, 5, 7, 11)]
M2

# Show relative importance of features
vip::vip(RFModel)

# Alternatively using ggplot function
var_imp <-varImp(RFModel)
ggplot(var_imp, aes(x = reorder(Variable, Importance), y = importance)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Feature Importance Plot for RF Model")

# Prepare data for explain() function
train$Malaria_Result1 <- ifelse(train$Malaria_Result == "Positive", 1, 0)
explainer <- explain(model = RFModel,
                     data = train[, -ncol(train)],  # Exclude the target column
                     y = train$Malaria_Result1,      # Target values as vector
                     label = "Random Forest")
# Generate feature importance
feature_importance <- model_parts(explainer)
plot(feature_importance)
```

```{r}
# Explain a single prediction
new_observation <- test[1, -ncol(test)]  # Select a test instance

# Break Down explanation
bd <- predict_parts(explainer, new_observation)
plot(bd)

# SHAP explanation
shap_values <- predict_parts(explainer, new_observation, type = "shap")
plot(shap_values)
```

```{r}
# Partial Dependence Plot for Temperature
pdp <- model_profile(explainer, variables = "Temperature")
plot(pdp)
```




