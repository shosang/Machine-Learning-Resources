# Fit a model with x (numerical) and z (categorical-factor) to predict y, along with the interaction between x and Z
lm(y ~ x + z + x:z, data = mydata)

# Scatter plot with interaction variables
mydata %>% ggplot(aes(x=x,y=y,color=z)) +
  geom_point() + 
  geom_smooth(method="lm", se=FALSE)