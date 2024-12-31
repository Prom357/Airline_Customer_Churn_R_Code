#### Set working directory
setwd(dirname(file.choose()))
getwd()

###### Import the CSV file ###########
airlinedata = read.csv(file.choose(), stringsAsFactors = FALSE)

# Install and load required packages
required_packages <- c("caret", "randomForest", "kknn", "naivebayes", "glmnet", 
                       "pROC", "PRROC", "ROSE", "reshape2", "psych", "dplyr", 
                       "rpart", "class", "rpart.plot", "pander", "nortest", "ggplot2")

new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)
lapply(required_packages, library, character.only = TRUE)

# Read data
airlinedata <- read.csv("airlinedata.csv", stringsAsFactors = FALSE)

# Inspect the data
head(airlinedata)
str(airlinedata)

# List of categorical variables
categorical_vars <- c("Sat", "Gender", "Cust_Type", "T_Travel", "Class", "Seat_C", 
                      "Depart_Arrival_C", "Food_drink", "Gate_loc", "wifi_service", 
                      "Inflight_ent", "Online_suprt", "Online_BK", "Onboard_service", 
                      "Leg._room", "Baggage_h", "Checkin", "Cleanliness", "Online_boarding")

# Convert categorical variables to factors
airlinedata[categorical_vars] <- lapply(airlinedata[categorical_vars], factor)
summary(airlinedata)

#### Data Visualization ##########
# Gender distribution
ggplot(airlinedata, aes(x = Gender, fill = factor(Gender, labels = c("Male", "Female")))) +
  geom_bar() +
  scale_fill_manual(values = c("blue", "red")) +
  labs(title = "Gender Distribution of Passengers", x = "Gender", y = "Count") +
  theme_minimal()

# Customer type distribution
ggplot(airlinedata, aes(x = Cust_Type, fill = factor(Cust_Type, labels = c("Disloyal", "Loyal")))) +
  geom_bar() +
  scale_fill_manual(values = c("red", "blue")) +
  labs(title = "Customer Type Distribution", x = "Customer Type", y = "Count") +
  theme_minimal()

# Age distribution
ggplot(airlinedata, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black") +
  labs(title = "Histogram of Passenger Ages", x = "Age", y = "Frequency")

# Satisfaction distribution
ggplot(airlinedata, aes(x = Sat, fill = factor(Sat, labels = c("Dissatisfied", "Satisfied")))) +
  geom_bar() +
  scale_fill_manual(values = c("red", "green")) +
  labs(title = "Satisfaction Distribution", x = "Satisfaction", y = "Count") +
  theme_minimal()

# Satisfaction vs. service (Box plots)
satisfaction_vars <- c("Seat_C", "Depart_Arrival_C", "Food_drink", "Gate_loc", "wifi_service", 
                       "Inflight_ent", "Online_suprt", "Online_BK", "Onboard_service", "Leg._room", 
                       "Baggage_h", "Checkin", "Cleanliness", "Online_boarding")
melted_data <- melt(airlinedata, id.vars = "Sat", measure.vars = satisfaction_vars, 
                    variable.name = "Service", value.name = "Satisfaction_Score")
ggplot(melted_data, aes(x = Service, y = Satisfaction_Score, fill = factor(Sat))) +
  geom_boxplot() +
  facet_wrap(~Service, scales = "free") +
  labs(x = "Service", y = "Satisfaction Score", title = "Box Plot of Satisfaction Scores by Service") +
  theme_minimal() +
  scale_fill_manual(values = c("red", "green"))

#### Correlation Analysis ##########
# Convert categorical variables to numeric
airlinedata[categorical_vars] <- lapply(airlinedata[categorical_vars], as.numeric)

# Calculate correlation matrix
numeric_vars <- c("Sat", "Age", "Flight_DS", "Seat_C", "Depart_Arrival_C", "Food_drink", 
                  "Gate_loc", "wifi_service", "Inflight_ent", "Online_suprt", "Online_BK", 
                  "Onboard_service", "Leg._room", "Baggage_h", "Checkin", "Cleanliness", 
                  "Online_boarding", "Depart_Delay_Min", "Arrival_Delay_Min")
correlation_matrix <- cor(airlinedata[, numeric_vars])
print(correlation_matrix)

# Create a heatmap of the correlation matrix
cor_df <- as.data.frame(as.table(correlation_matrix))
ggplot(cor_df, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "yellow", na.value = "white") +
  labs(title = "Correlation Heatmap", x = "Variable 1", y = "Variable 2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

############################ Modelling Section ##################################

# Load necessary libraries
library(ggplot2)
library(caret)
library(class)
library(naivebayes)
library(rpart)
library(rpart.plot)
library(randomForest)

############################ Data Partition #####################################

# Change categorical dependent variable to factor
airlinedata$Sat <- as.factor(airlinedata$Sat)

# Split the data into training and test sets (70% train, 30% test)
set.seed(123)
indep2 <- sample(2, nrow(airlinedata), replace = TRUE, prob = c(0.7, 0.3))

train <- airlinedata[indep2 == 1, ]
test <- airlinedata[indep2 == 2, ]

######################## Logistic Regression ####################################

# Train the logistic regression models
model <- glm(Sat ~ Seat_C + Depart_Arrival_C + Food_drink + Gate_loc +
               wifi_service + Inflight_ent + Online_suprt + Online_BK +
               Onboard_service + Leg._room + Baggage_h + Checkin +
               Cleanliness + Online_boarding + Depart_Delay_Min +
               Arrival_Delay_Min + Gender + Age + Cust_Type + T_Travel +
               Class + Flight_DS, 
             data = train, family = "binomial")
summary(model)

model2 <- glm(Sat ~ Seat_C + Depart_Arrival_C + Food_drink + Gate_loc +
                Inflight_ent + Online_suprt + Online_BK + Onboard_service +
                Leg._room + Baggage_h + Checkin + Cleanliness +
                Online_boarding + Arrival_Delay_Min + Gender + Age +
                Cust_Type + T_Travel + Class + Flight_DS, 
              data = train, family = "binomial")
summary(model2)

# Predictions and Confusion Matrix
predict_train <- predict(model, train, type = "response")
predict_train <- factor(ifelse(predict_train > 0.5, 1, 0), levels = c(0, 1))
confusionMatrix(predict_train, train$Sat)

predict_test <- predict(model, test, type = "response")
predict_test <- factor(ifelse(predict_test > 0.5, 1, 0), levels = c(0, 1))
confusionMatrix(predict_test, test$Sat)

predict_test2 <- predict(model2, test, type = "response")
predict_test2 <- factor(ifelse(predict_test2 > 0.5, 1, 0), levels = c(0, 1))
confusionMatrix(predict_test2, test$Sat)

# Plot Logistic Regression Performance
log_metrics <- data.frame(
  Metric = c("Sensitivity", "Specificity", "Pos Pred Value", "Neg Pred Value", "Accuracy"),
  Value = c(0.8837, 0.7970, 0.7835, 0.8918, 0.8364)
)

ggplot(log_metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.4f", Value)), vjust = -0.3) +
  labs(title = "Logistic Regression Model Evaluation Performance", y = "Value") +
  ylim(0, 1) +
  theme_minimal() +
  theme(legend.position = "none")

############################### K-Nearest Neighbors ############################

# Feature Scaling
predictor_vars <- c("Seat_C", "Depart_Arrival_C", "Food_drink", "Gate_loc",
                    "wifi_service", "Inflight_ent", "Online_suprt", "Online_BK",
                    "Onboard_service", "Leg._room", "Baggage_h", "Checkin",
                    "Cleanliness", "Online_boarding", "Arrival_Delay_Min",
                    "Gender", "Age", "Cust_Type", "T_Travel", "Class", "Flight_DS")

train_scaled <- as.data.frame(scale(train[, predictor_vars]))
test_scaled <- as.data.frame(scale(test[, predictor_vars]))

# Train KNN Model
k <- 5
knn_model <- knn(train = train_scaled, test = test_scaled, cl = train$Sat, k = k)
confusionMatrix(knn_model, test$Sat)

# Adjusted KNN Model
k <- 9
knn_model2 <- knn(train = train_scaled, test = test_scaled, cl = train$Sat, k = k)
confusionMatrix(knn_model2, test$Sat)

# Plot KNN Performance
knn_metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value"),
  Value = c(0.9267, 0.9402, 0.9154, 0.9024, 0.9485)
)

ggplot(knn_metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 4)), vjust = -0.2, color = "black") +
  labs(title = "K-Nearest Neighbors Model Performance", x = "Metrics", y = "Values") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

############################ Naive Bayes #######################################

# Train Naive Bayes Model
nb_model <- naive_bayes(Sat ~ ., data = train[, c("Sat", predictor_vars)])
nb_predictions <- predict(nb_model, newdata = test)
confusionMatrix(nb_predictions, test$Sat)

# Plot Naive Bayes Performance
nb_metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "Negative Predictive Value"),
  Value = c(0.8289, 0.8155, 0.8401, 0.8092, 0.8456)
)

ggplot(nb_metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Value, 3)), vjust = -0.2, color = "black") +
  labs(title = "Naive Bayes Model Performance", x = "Metrics", y = "Values") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

############################ Decision Tree #####################################

# Train and Evaluate Decision Tree
dt_model <- rpart(Sat ~ ., data = train, method = "class")
dt_predictions <- predict(dt_model, newdata = test, type = "class")
confusionMatrix(dt_predictions, test$Sat)

# Plot Decision Tree
rpart.plot(dt_model)

############################ Random Forest #####################################

# Train Random Forest Model
rf_model <- randomForest(Sat ~ ., data = train)
rf_predictions <- predict(rf_model, newdata = test)
confusionMatrix(rf_predictions, test$Sat)


#####Compare the models if needed### Optional
