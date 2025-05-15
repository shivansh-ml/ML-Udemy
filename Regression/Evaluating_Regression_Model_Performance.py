'''
For regression problems only
R Squared can indeed be negative when the model fits worse than a horizontal line representing the mean, indicating a poor model quality. 

R^2 = 1 - SS res / SS total

Adjusted R squared = 1 - (1-R^2) - (n-1)/(n-k-1)

k - no of regressor/Independent variable
n - sample size
'''