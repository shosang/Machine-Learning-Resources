# Fit a logistic regression model
mod <- glm(y ~ x, data = mydata, family = binomial)

# Make an out-of-sample prediction
new_data <- data.frame(x=x1)
augment(mod, newdata = new_data, type.predict = "response")

# response variable is binary but fitted values are probabilities
# Use 0.5 as the threshold for making binary decisions
mod_tidy <- augment(mod, type.predict="response") %>%
  mutate(y_hat = round(.fitted))

# Create a confusion matrix of the results
mod_tidy <-select(y, y_hat) %>%
  table()