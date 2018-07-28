## Interaction ##

# Fit a model with x (numerical) and z (categorical-factor) to predict y, along with the interaction between x and Z
lm(y ~ x + z + x:z, data = mydata)

# Scatter plot with interaction variables
mydata %>% ggplot(aes(x=x,y=y,color=z)) +
  geom_point() + 
  geom_smooth(method="lm", se=FALSE)

## -----------------------------------------------------------------------------
## Numerical Variables ##

# Fit a model with 2 numerical vairables: x1 and x2
mod <- lm(y ~ x1 + x2, data=mydata)

## HeatMap
# 1. Create grid of all possible pairs of values
grid <- mydata %>%
  data_grid(
    x1 = seq_range(x1, by=1),
    x2 = seq_range(x2, by=2)
  )

# 2. Augment to find the fitted values corresponding to grid
aug <- augment(mod, grid)

# 3. Tiles in data space - heat map
mydata %>% ggplot(aes(x=x1,y=x2)) +
  geom_point(aes(color=y)) +
  geom_tile(data=aug, aes(fill=.fitted, alpha=0.5)) 

## 3D Scatterplot using plotly for z as a function of x and y with factor w
plot_ly(data=mydata, z=~z, x=~x, y=~y, opacity=0.6) %>%
  add_markers(color=~w)



