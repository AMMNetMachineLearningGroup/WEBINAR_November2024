##################################################################
##################################################################
########### EXPLAINABLE AI FOR MALARIA MODELING ####################
##################################################################
############ INSTRUCTOR: OLAWALE AWE, PhD ########################
############### AMMnet ML GROUP ################################
############## ###################################################
############## 26TH OCTOBER, 2024 #################################
##################################################################

#Definition: 
#Explainable AI (XAI)refers to a set of tools and techniques that 
#make the behavior of machine learning models more interpretable and 
#understandable to humans.

#Goals
#The goal of XAI is to clarify how and why AI models make certain predictions
#ensuring that both experts and non-experts can trust and effectively use these 
#insights in real-world applications.

#In malaria modeling, for instance, XAI can explain why an AI model predicts 
#high malaria severity in revealing which symptoms (like vomiting, jundice, or fever) are influencing that prediction. 
#By increasing model interpretability, XAI helps bridge the gap between complex AI technologies and actionable insights 
#for policymakers, healthcare workers, and researchers, allowing them to use AI insights more confidently in planning and intervention.

#TYpes
#1.SHapleY Additive Model exPlanations (SHAP)
#2.Local Interpretable Model-Agnostic Explanations (LIME)

#These methods help identify which features (or variables) contribute most to a model’s predictions.

# Load necessary packages
#install.packages("caret")  # If not already installed
#install.packages("iml")
#install.packages("lime")
#install.packages('tictoc')
#install.packages("ggplot2")
#install.packages("dplyr")

#Load LIBRARIES
library(caret)
library(iml)
library(lime)
library(tictoc)
library(ggplot2)
library(dplyr)


######Load the Malaria dataset
###Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7093799/

mdata = read.csv("Malaria-Data.csv", header = TRUE)
mdata
dim(mdata)
mdata
head(mdata)
names(mdata)
#str(odata)
#attach(mdata)
summary(mdata) ###Descriptive Statistics
#describe(mdata)###Descriptive Statistics
sum(is.na(mdata))###Check for missing data

mdata=mdata[,-1] ##Exclude Age
names(mdata)

#Perform Featureplot to see the data distribution at a glance
featurePlot(x = mdata[, -which(names(mdata) == "severe_maleria")],   # Predictors
           y = mdata$severe_maleria,                               # Target variable
           plot = "box",                                       # Type of plot (e.g., "box", "density", "scatter")
          strip = strip.custom(strip.names = TRUE),            # Add strip labels
          scales = list(x = list(relation = "free"),           # Scales for x-axis
                    y = list(relation = "free")))  


###Rename the classes of the Target variable and plot it to determine imbalance
mdata$severe_maleria <- factor(mdata$severe_maleria, 
                               levels = c(0,1), 
                               labels = c('Not Severe', 'Severe'))
###Plot Target Variable
plot(factor(severe_maleria), names= c('Not Severe', 'Severe'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='Malaria Diagnosis')
box()
#Or use ggplot 
ggplot(mdata, aes(x = factor(severe_maleria))) + geom_bar() + labs(x = "Malaria Detected", y = "Count")
################################################################

###VIEW THE AVAILABLE MODELS IN CARET
models = getModelInfo()
names(models)


#TODAY we are going to train the following machine learning models:
#######################################################
#LR
#SVM
#RANDOM FOREST
#NAIVE BAYES
#KNN
#LDA
#NNET/mlp
#LVQ
#Bagging
#Boosting
#DT

#STEPS
#1. Data Preparation and Preprocessing
# --Cleaning, Feature Engineering,Visualization, Data Splitting, etc
#2. Define the Training Control- Set up cross validation
#3. Train the Models- Select the ML models you want to train 
#4. Evaluate your model using test data
#5. Tune the hyperparameters and Resample the data (optional)
#6. Implement XAI

###DATA PARTITION FOR MACHINE LEARNING
##################################################################
#caret can also be used for data partition
set.seed(123)
trainIndex <- createDataPartition(mdata$severe_maleria, p = 0.7, list = FALSE)
train <- mdata[trainIndex, ]
test <- mdata[-trainIndex, ]
dim(train)
dim(test)

# Set seed for reproducibility
set.seed(123)

# Define control for training
#control1 <- trainControl(method = "cv", number = 10)
control1 <- trainControl(method ="repeatedcv", number=10, repeats=5, sampling='smote', search='random')## For tuning

# 1. Train the Logistic Regression Model
tic()
lrModel <- train(factor(severe_maleria) ~ ., data = train, method = "glm", trControl = control1)
toc()

# Predict on the test set
lrpred <- predict(lrModel, newdata = test)

# Evaluate with Confusion Matrix
lr.cM <- confusionMatrix(lrpred, as.factor(test$severe_maleria), positive = "Severe", mode = "everything")
print(lr.cM)
# Plotting confusion matrix
fourfoldplot(lr.cM$table, col = rainbow(4), main = "LR Confusion Matrix")
plot(varImp(lrModel, scale = TRUE))


# ---------- SHAP (SHapley Additive exPlanations) ----------

#The Shapley value helps explain how much each feature contributes to the prediction made by a machine learning model. 
#It provides a way to fairly distribute the "credit" for the model’s output across all input features.

#By visualizing the SHAP plot, you can understand not only which features are important, 
#but also how specific feature values that are driving predictions for individual cases.

# Set seed for reproducibility
set.seed(456)

# Assuming lrModel is already trained :
# Convert the caret model to a Predictor object, separating the target variable
predictorlr <- Predictor$new(lrModel, data = train[, -which(names(train) == "severe_maleria")], y = train$severe_maleria)

# Select a single instance from the test set to explain
# Replace '1' with the index of any other instance if desired
x_interest <- test[1, -which(names(test) == "severe_maleria")]

# Compute SHAP values for the specific instance
shapleylr <- Shapley$new(predictorlr, x.interest = x_interest)

# Plot the SHAP values for this instance
shapleylr$plot() + ggtitle("SHAP Values for a Single Instance in Logistic Regression Model")

########INTERPRETATION
#Each feature has its own SHAP value, calculated in the context of all other features.
#The direction and length of the bar indicate the magnitude and impact on the prediction.

#Rightward (positive): Indicates the feature is pushing the model prediction towards a positive class (e.g., "Severe" if that is the positive label).


##RANDOM FOREST
# 2. Train the Random Forest Classifier
tic()
rfModel <- train(factor(severe_maleria) ~ ., data = train, method = "rf", trControl = control1)
toc()

# Predict on the test set
rfpred <- predict(rfModel, newdata = test)

# Evaluate with Confusion Matrix
rf.cM <- confusionMatrix(rfpred, as.factor(test$severe_maleria), positive = "Severe", mode = "everything")
print(rf.cM)
# Plotting confusion matrix
fourfoldplot(rf.cM$table, col = rainbow(4), main = "RF Confusion Matrix")
plot(varImp(rfModel, scale = TRUE))


# ---------- SHAP (SHapley Additive exPlanations) ----------

#The Shapley value helps explain how much each feature contributes to the prediction made by a machine learning model. 
#It provides a way to fairly distribute the "credit" for the model’s output across all input features.

#By visualizing the SHAP plot, you can understand not only which features are important, 
#but also how specific feature values that are driving predictions for individual cases.

# Set seed for reproducibility
set.seed(456)

# Assuming lrModel is already trained :
# Convert the caret model to a Predictor object, separating the target variable
predictorrf <- Predictor$new(rfModel, data = train[, -which(names(train) == "severe_maleria")], y = train$severe_maleria)

# Select a single instance from the test set to explain
# Replace '1' with the index of any other instance if desired
x_interest <- test[1, -which(names(test) == "severe_maleria")]

# Compute SHAP values for the specific instance
shapleyrf <- Shapley$new(predictorrf, x.interest = x_interest)

# Plot the SHAP values for this instance
shapleyrf$plot() + ggtitle("SHAP Values for a Single Instance in Random Forest Model")

#Leftward (negative): Indicates the feature is pushing the model prediction towards a negative class (e.g., "Not Severe").

#Larger absolute SHAP values mean a feature has a stronger influence on the prediction.
#Smaller SHAP values (close to zero) indicate that a feature has minimal influence on the model's output for that instance

#TRY FOR OTHER MODELS
#NB
#DT
#NN
#KNN
#SVM
