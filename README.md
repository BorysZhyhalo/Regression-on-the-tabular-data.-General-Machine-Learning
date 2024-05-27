# Regression-on-the-tabular-data.-General-Machine-Learning
Basic Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 90000 entries, 0 to 89999
Data columns (total 54 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 53  target  90000 non-null  float64
dtypes: float64(43), int64(11)
memory usage: 37.1 MB
None

We have a data set with dtypes: float64(43), int64(11).
Out target variable is 'target'

* Skewness: -0.0042 suggests the data is almost symmetric.
* Kurtosis: -1.2015 indicates that the data distribution is flatter and has lighter tails compared to a normal distribution.
* These values together suggest that data is roughly symmetric but has a flatter shape than a typical bell curve.

## The scatter plot shown the relationship between the feature labeled "8" and "6" and the target variable. Here are a few key points to understand from this scatter plot:

* Binary Feature: The feature "8" appears to be binary, taking only two distinct values, 0 and 1. This is evident from the two vertical lines of points at x=0 and x=1.
* Parabolic Relationship: The plot shows a clear parabolic (U-shaped) relationship between the feature "6" and the target variable. This means that as the feature "6" moves away from 0 in either direction (negative or positive), the target value increases.

  ### Next Steps
* Log(target)=n*log(x6)
* This is a linear model
* It can be estimated by linear regression

Let's build Linear Regression model

# Explanation of the Script:
### Data Loading and Transformation:

* Load the dataset.
* Transform the target and feature 6 using logarithms. To avoid issues with log(0), add 1 to the feature values before taking the log.

### Data Splitting:
* Split the data into training and testing sets using an 80-20 split.

### Model Building:
* Build a linear regression model using the transformed feature (log_feature_6) and transformed target (log_target).

### Prediction and Transformation:
* Make predictions on the test set.
* Transform the predicted and actual values back to the original scale using the exponential function.

### Model Evaluation:
* Evaluate the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) or R2.
* Print the evaluation metrics and model coefficients.

Our output
Mean Squared Error (MSE): 1.7587525945135074
Root Mean Squared Error (RMSE): 1.3261796991786248
R2: 0.9979087345778254

### Try polynomial regression for better fit
Explanation of the Script:
Data Loading and Transformation:

Load the dataset.
Transform the target and feature 6 using logarithms. To avoid issues with log(0), add 1 to the feature values before taking the log.
Data Splitting:

Split the data into training and testing sets using an 80-20 split.
Polynomial Feature Creation:

Use PolynomialFeatures to create polynomial features of the specified degree (in this case, degree 2).
Model Building:

Build a polynomial regression model using the transformed polynomial features and transformed target.
Prediction and Transformation:

Make predictions on the test set.
Transform the predicted and actual values back to the original scale using the exponential function.
Model Evaluation:

Evaluate the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
Print the evaluation metrics and model coefficients.

Polynomial Regression (degree 2):
Mean Squared Error (MSE): 0.6829706737781283
Root Mean Squared Error (RMSE): 0.8264203977263196
R2: 0.9991879057015256


### The results indicate that the Polynomial Regression model with a degree of 2 provides a better results to Linear Regression model. Here are the steps and outputs for clarity:

## Linear Regression Results:
* Mean Squared Error (MSE): 1.7587525945135074
* Root Mean Squared Error (RMSE): 1.3261796991786248

## Polynomial Regression Results:
* Mean Squared Error (MSE): 0.6829706737781283
* Root Mean Squared Error (RMSE): 0.8264203977263196

## Interpretation:
* The Polynomial Regression model (degree 2) significantly reduces both the MSE and RMSE compared to the Linear Regression model.
* This improvement suggests that the relationship between the feature 6 and the target is better captured by the polynomial model.
* Comparing performance of the polynomial model and the linear regression model we can see that the output is the same.


  
