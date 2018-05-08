## Deep learning with Keras: Chest accelerometer data preparation


library(readr)
library(dplyr)

# Files from the download are in a directory called "activity" saved in the same place as this script
data_files <- list.files("activity", pattern = "*.csv", full.names = TRUE)

# Initialise array for data:
#   Rows will be observations
#   Columns will be:
#   * Time point (sequential count integer)
#   * x-, y-, z-directional accelerometer data time series (integer)
#   * Activity label (1-7)
#   * Person label (0-14)
dataset <- data.frame()

# Add data from each file in turn
for (k in seq_along(data_files)) {
  
  cat("Reading file", k, "/", length(data_files), "\n")
  
  d <- read_csv(data_files[k], col_names = c("obs", "acc_x", "acc_y", "acc_z", "activity"), col_types = "ddddd")
  
  # Add a column with a label representing the person
  d$person <- k
  
  dataset <- bind_rows(dataset, d)
}


# Reshape data into 3 dimensions:
#   1-dimension ("rows") is observations (1926896 in total)
#   2-dimension ("columns") is time series values (260 = 5{seconds}*52{Hz} in total)
#   3-dimension ("leaves") are as follows (5 in total):
#     * 3 directions (x-, y-, z-acceleration)
#     * Activity type labels
#     * Person labels
#
# We'll chop the time series into 260-length (5 second) sections every 52 points (every 1 second)
m <- (nrow(dataset) - 208) %/% 52
chopped <- array(0, dim = c(m, 260, 5))

for (k in seq_len(m)) {
  
  start <- 52*(k-1) + 1
  stop <- start + 259
  
  # If the count column's value at "stop" is smaller than at "start", we've changed person, so discard
  # If the activity label column is not all the same, we have more than one activity in that section, so discard
  if (dataset[stop, "obs"] < dataset[start, "obs"] ||
      !all(dataset[start:stop, "activity"] == dataset[start, "activity"])) {
    next
  }
  
  # Else copy all but count column to the new data block
  chopped[k, , ] <- as.matrix(dataset[start:stop, -1])
}

# Remove the extra rows, which will have person label 0
chopped <- chopped[(chopped[, 1, 5] != 0), , ]

# "walking" corresponds to activity label 4
walking <- chopped[(chopped[, 1, 4] == 4), , ]


# Scale each time series individually, because recorded data is not necessarily calibrated
walking_data <- apply(walking[, , 1:3], c(1, 3), scale)

# The person label is in layer 5, and it's the same in all columns so we just get it from column 1
walking_labels <- walking[, 1, 5]

# Check shapes
dim(walking_data)
dim(walking_labels)


# Save this information in .rds files
saveRDS(walking_data, "walking_data.rds")
saveRDS(walking_labels, "walking_labels.rds")
