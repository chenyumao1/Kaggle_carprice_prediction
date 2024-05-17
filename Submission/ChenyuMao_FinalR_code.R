setwd("/Users/maocheny/Desktop/Book/Columbia-研究生/APAN 5200")

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

data = read.csv('analysisData.csv')
scoring = read.csv('scoringData.csv')

library(caret)
library(car)
library(tidyr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(broom)
library(xgboost)

sapply(data, class)
sum(is.na(data))

selected_data <- data %>% select_if(is.numeric)
names(selected_data)

selected_data$fuel_tank_volume_gallons[is.na(selected_data$fuel_tank_volume_gallons)] <- median(selected_data$fuel_tank_volume_gallons, na.rm = TRUE)
selected_data$highway_fuel_economy[is.na(selected_data$highway_fuel_economy)] <- median(selected_data$highway_fuel_economy, na.rm = TRUE)
selected_data$city_fuel_economy[is.na(selected_data$city_fuel_economy)] <- median(selected_data$city_fuel_economy, na.rm = TRUE)
selected_data$wheelbase_inches[is.na(selected_data$wheelbase_inches)] <- median(selected_data$wheelbase_inches, na.rm = TRUE)
selected_data$back_legroom_inches[is.na(selected_data$back_legroom_inches)] <- median(selected_data$back_legroom_inches, na.rm = TRUE)
selected_data$front_legroom_inches[is.na(selected_data$front_legroom_inches)] <- median(selected_data$front_legroom_inches, na.rm = TRUE)
selected_data$length_inches[is.na(selected_data$length_inches)] <- median(selected_data$length_inches, na.rm = TRUE)
selected_data$width_inches[is.na(selected_data$width_inches)] <- median(selected_data$width_inches, na.rm = TRUE)
selected_data$height_inches[is.na(selected_data$height_inches)] <- median(selected_data$height_inches, na.rm = TRUE)
selected_data$engine_displacement[is.na(selected_data$engine_displacement)] <- median(selected_data$engine_displacement, na.rm = TRUE)
selected_data$horsepower[is.na(selected_data$horsepower)] <- median(selected_data$horsepower, na.rm = TRUE)
selected_data$daysonmarket[is.na(selected_data$daysonmarket)] <- median(selected_data$daysonmarket, na.rm = TRUE)
selected_data$maximum_seating[is.na(selected_data$maximum_seating)] <- median(selected_data$maximum_seating, na.rm = TRUE)
selected_data$year[is.na(selected_data$year)] <- median(selected_data$year, na.rm = TRUE)
selected_data$mileage[is.na(selected_data$mileage)] <- median(selected_data$mileage, na.rm = TRUE)
selected_data$owner_count[is.na(selected_data$owner_count)] <- median(selected_data$owner_count, na.rm = TRUE)
selected_data$seller_rating[is.na(selected_data$seller_rating)] <- median(selected_data$seller_rating, na.rm = TRUE)

selected_data <- selected_data[, -which(names(selected_data) == "id")]
sapply(selected_data, class)
sum(is.na(selected_data))

set.seed(100)
index <- createDataPartition(selected_data$price, p= 0.8, list=FALSE)
train <- selected_data[index, ]
test <- selected_data[-index, ]

dtrain <- xgb.DMatrix(data = as.matrix(train[, -which(names(train) == "price")]), label = train$price)
dtest <- xgb.DMatrix(data = as.matrix(test[, -which(names(test) == "price")]), label = test$price)

params <- list(
  booster = "gbtree",
  objective = "reg:squarederror"
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 10000,
  watchlist = list(eval = dtest, train = dtrain),
  print_every_n = 10,
  early_stopping_rounds = 500,
  maximize = FALSE
)

pred_1 = predict(xgb_model,
                 newdata = dtest)

rmse_1 = rmse(test$price, pred_1)
rmse_1

params_grid <- expand.grid(
  nrounds = 100, 
  eta = c(0.01, 0.05, 0.1),
  max_depth = c(8, 9),
  gamma = 0,
  min_child_weight = c(2, 3),
  subsample = c(0.75, 1),
  colsample_bytree = c(0.75, 1)
)

train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  allowParallel = TRUE
)

xgb_model <- train(
  price ~ .,
  data = train, 
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = params_grid,
  metric = "RMSE"
)

best_params <- xgb_model$bestTune
best_params

params_final <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.1,
  gamma = 0, 
  min_child_weight = 2,
  max_depth = 9,
  subsample = 1, 
  colsample_bytree = 0.75
)

xgb_model <- xgb.train(
  params = params_final,
  data = dtrain,
  nrounds = 10000,
  watchlist = list(eval = dtest, train = dtrain),
  print_every_n = 10,
  early_stopping_rounds = 5000,
  maximize = FALSE
)

pred_2 = predict(xgb_model,
                 newdata = dtest)

rmse_2 = rmse(test$price, pred_2)
rmse_2

dtrain_final <- xgb.DMatrix(data = as.matrix(selected_data[, -which(names(selected_data) == "price")]), 
                            label = selected_data$price)

params_final <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.1,
  gamma = 0, 
  min_child_weight = 2,
  max_depth = 9,
  subsample = 1, 
  colsample_bytree = 0.75
)

xgb_model_final <- xgb.train(
  params = params_final,
  data = dtrain_final,
  nrounds = 10000,
  watchlist = list(eval = dtest, train = dtrain),
  print_every_n = 10,
  early_stopping_rounds = 5000,
  maximize = FALSE
)

pred_3 = predict(xgb_model_final,
                 newdata = dtest)

rmse_3 = rmse(test$price, pred_3)
rmse_3

selected_scoring <- scoring %>% select_if(is.numeric)
selected_scoring$fuel_tank_volume_gallons[is.na(selected_scoring$fuel_tank_volume_gallons)] <- median(selected_scoring$fuel_tank_volume_gallons, na.rm = TRUE)
selected_scoring$highway_fuel_economy[is.na(selected_scoring$highway_fuel_economy)] <- median(selected_scoring$highway_fuel_economy, na.rm = TRUE)
selected_scoring$city_fuel_economy[is.na(selected_scoring$city_fuel_economy)] <- median(selected_scoring$city_fuel_economy, na.rm = TRUE)
selected_scoring$wheelbase_inches[is.na(selected_scoring$wheelbase_inches)] <- median(selected_scoring$wheelbase_inches, na.rm = TRUE)
selected_scoring$back_legroom_inches[is.na(selected_scoring$back_legroom_inches)] <- median(selected_scoring$back_legroom_inches, na.rm = TRUE)
selected_scoring$front_legroom_inches[is.na(selected_scoring$front_legroom_inches)] <- median(selected_scoring$front_legroom_inches, na.rm = TRUE)
selected_scoring$length_inches[is.na(selected_scoring$length_inches)] <- median(selected_scoring$length_inches, na.rm = TRUE)
selected_scoring$width_inches[is.na(selected_scoring$width_inches)] <- median(selected_scoring$width_inches, na.rm = TRUE)
selected_scoring$height_inches[is.na(selected_scoring$height_inches)] <- median(selected_scoring$height_inches, na.rm = TRUE)
selected_scoring$engine_displacement[is.na(selected_scoring$engine_displacement)] <- median(selected_scoring$engine_displacement, na.rm = TRUE)
selected_scoring$horsepower[is.na(selected_scoring$horsepower)] <- median(selected_scoring$horsepower, na.rm = TRUE)
selected_scoring$daysonmarket[is.na(selected_scoring$daysonmarket)] <- median(selected_scoring$daysonmarket, na.rm = TRUE)
selected_scoring$maximum_seating[is.na(selected_scoring$maximum_seating)] <- median(selected_scoring$maximum_seating, na.rm = TRUE)
selected_scoring$year[is.na(selected_scoring$year)] <- median(selected_scoring$year, na.rm = TRUE)
selected_scoring$mileage[is.na(selected_scoring$mileage)] <- median(selected_scoring$mileage, na.rm = TRUE)
selected_scoring$owner_count[is.na(selected_scoring$owner_count)] <- median(selected_scoring$owner_count, na.rm = TRUE)
selected_scoring$seller_rating[is.na(selected_scoring$seller_rating)] <- median(selected_scoring$seller_rating, na.rm = TRUE)

selected_scoring <- selected_scoring[, -which(names(selected_scoring) == "id")]
sapply(selected_scoring, class)
sum(is.na(selected_scoring))

pred_final = predict(xgb_model_final, newdata = as.matrix(selected_scoring))
submissionFile_final <- data.frame(id = scoring$id, price = pred_final)
write.csv(submissionFile_final, 'chenyumao_kaggle_submission_final.csv',row.names = F)
result <- read.csv("chenyumao_kaggle_submission_final.csv")
nrow(result)
sum(is.na(result))


