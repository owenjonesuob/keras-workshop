# See deep_learning_with_keras_r.Rmd for explanations of everything!

set.seed(2018)



# Import packages we'll be using
library(keras)



# Prepare data
iris

iris <- iris[sample(nrow(iris)), ]
iris_labels <- as.numeric(iris$Species) - 1

iris_onehot <- to_categorical(iris_labels)
iris_onehot



# Build the model
model <- keras_model_sequential()

model %>%
  layer_dense(15, activation = "sigmoid", input_shape = 4) %>%
  layer_dense(3, activation = "softmax")

summary(model)



# Compile and fit
model %>%
  compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = c("accuracy"))

model %>%
  fit(as.matrix(iris[, -5]), iris_onehot, epochs = 50, batch_size = 20, validation_split = 0.2)



# Load walking data
walking <- readRDS("data/walking_data.rds")
dim(walking)



# Set up a quick plotting function
plot_series <- function(series) {
  # x-channel
  plot(series[, 1], type = "l", col = "red")
  # y-channel
  lines(series[, 2], col = "darkgreen")
  # z-channel
  lines(series[, 3], col = "blue")
}


plot_series(walking[100, , ])



# Import labels (they're stored separately!)
walking_labels <- readRDS("data/walking_labels.rds")
unique(walking_labels)



# Prepare partitions for training/testing
m <- nrow(walking)

indices <- sample(1:m, m)

train_indices <- indices[1:floor(m*0.6)]
val_indices <- indices[ceiling(m*0.6):floor(m*0.8)]
test_indices <- indices[ceiling(m*0.8):m]

X_train <- walking[train_indices, , ]
X_val <- walking[val_indices, , ]
X_test <- walking[test_indices, , ]

y_train <- to_categorical(walking_labels[train_indices])
y_val <- to_categorical(walking_labels[val_indices])
y_test <- to_categorical(walking_labels[test_indices])



# Create the model
model <- keras_model_sequential()

model %>%
  layer_conv_1d(filters = 30, kernel_size = 40, strides = 2, activation = "relu", input_shape = c(260, 3))

model %>%
  layer_max_pooling_1d(pool_size = 2)

model %>%
  layer_conv_1d(filters = 40, kernel_size = 10, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 2)



# Check model output
model$output_shape

model %>%
  layer_flatten()

model$output_shape


model %>%
  layer_dense(units = 100, activation = "sigmoid") %>%
  layer_dense(units = 15, activation = "softmax")



# See what we've done!
summary(model)



# Compile and fit
model %>%
  compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("accuracy"))

model %>%
  fit(X_train, y_train, epochs = 10, batch_size = 100, validation_data = list(X_val, y_val))



# Prediction and reporting
y_pred <- model %>%
  predict_classes(X_test)

table("Actual" = max.col(y_test) - 1, "Predicted" = y_pred)



# Looking inside the neurons
plot_filter <- function(model, layer, k) {
  weights <- get_weights(model$layers[[layer]])[[1]][, , k]
  plot_series(weights)
}


model %>%
  plot_filter(1, 5)



# Autocorrelation of learned weights
plot_filter_corr <- function(model, layer, k) {
  
  weights <- get_weights(model$layers[[layer]])[[1]][, , k]
  
  corrs <- apply(weights, 2, function(x) {
    acf(x, lag.max = nrow(weights), plot = FALSE)[["acf"]]
  })
  
  plot_series(corrs)
}


model %>%
  plot_filter_corr(1, 5)