import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

#Explanation of the Script:
#Data Loading and Transformation:

#Load the dataset.
#Transform the target and feature 6 using logarithms. To avoid issues with log(0), add 1 to the feature values before taking the log.

#Data Splitting:
#Split the data into training and testing sets using an 80-20 split.

#Model Building:
##Build a linear regression model using the transformed feature (log_feature_6) and transformed target (log_target).
#Prediction and Transformation:

##Make predictions on the test set.

#Transform the predicted and actual values back to the original scale using the exponential function.
#Model Evaluation:

#Evaluate the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
#Print the evaluation metrics and model coefficients.

df = pd.read_csv('train.csv')

# Transform the target and feature 6 using logarithms
df['log_target'] = np.log(df['target'])
df['log_feature_6'] = np.log(np.abs(df['6']) + 1)  # Adding 1 to avoid log(0)

# Split the df into training and testing sets
X = df[['log_feature_6']]
y = df['log_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred_log = model.predict(X_test)

# Transform predictions back to original scale
y_test_exp = np.exp(y_test)
y_pred_exp = np.exp(y_pred_log)

# Evaluate the model using MSE and RMSE
mse = mean_squared_error(y_test_exp, y_pred_exp)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_exp, y_pred_exp)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2:", r2)

# Plot the comparison using only column 6
plt.scatter(y_test_exp, y_pred_exp, label='True Target', alpha=0.5)
plt.xlabel('y_test_exp_poly')
plt.ylabel('y_pred_exp_poly')
plt.title('True Target vs Predicted Target Using Column 6')
plt.legend()
plt.show()

# Summary of the model
print("\nModel Coefficients:")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

#Try polynomial regression for better fit
#Explanation of the Script:
#Data Loading and Transformation:

#Load the dataset.
#Transform the target and feature 6 using logarithms. To avoid issues with log(0), add 1 to the feature values before taking the log.
#Data Splitting:
#Split the data into training and testing sets using an 80-20 split.

#Polynomial Feature Creation:
#Use PolynomialFeatures to create polynomial features of the specified degree (in this case, degree 2).

#Model Building:
#Build a polynomial regression model using the transformed polynomial features and transformed target.

#Prediction and Transformation:
#Make predictions on the test set.
#Transform the predicted and actual values back to the original scale using the exponential function.
#Model Evaluation:

#Evaluate the model using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

#Print the evaluation metrics and model coefficients.

# Transform the target and feature 6 using logarithms
df['log_target'] = np.log(df['target'])
df['log_feature_6'] = np.log(np.abs(df['6']) + 1)  # Adding 1 to avoid log(0)

# Split the df into training and testing sets
X = df[['log_feature_6']]
y = df['log_target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
degree = 2  # You can change the degree for higher order polynomials
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Build the Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predict on the test set
y_pred_log_poly = poly_model.predict(X_test_poly)

# Transform predictions back to original scale
y_test_exp_poly = np.exp(y_test)
y_pred_exp_poly = np.exp(y_pred_log_poly)

# Evaluate the model using MSE and RMSE
mse_poly = mean_squared_error(y_test_exp_poly, y_pred_exp_poly)
rmse_poly = np.sqrt(mse_poly)
r2 = r2_score(y_test_exp_poly, y_pred_exp_poly)

print("Polynomial Regression (degree {}):".format(degree))
print("Mean Squared Error (MSE):", mse_poly)
print("Root Mean Squared Error (RMSE):", rmse_poly)
print("R2:", r2)

# Plot the comparison using only column 6
plt.scatter(y_test_exp_poly, y_pred_exp_poly, label='True Target', alpha=0.5)
plt.xlabel('y_test_exp_poly')
plt.ylabel('y_pred_exp_poly')
plt.title('True Target vs Predicted Target Using Column 6')
plt.legend()
plt.show()

# Summary of the model
print("\nModel Coefficients:")
print("Intercept:", poly_model.intercept_)
print("Coefficient:", poly_model.coef_)

### The results indicate that the Polynomial Regression model with a degree of 2 provides a better results to Linear Regression model. Here are the steps and outputs for clarity:

## Linear Regression Results:
# Mean Squared Error (MSE): 1.7587525945135074
# Root Mean Squared Error (RMSE): 1.3261796991786248

## Polynomial Regression Results:
# Mean Squared Error (MSE): 0.6829706737781283
# Root Mean Squared Error (RMSE): 0.8264203977263196

## Interpretation:
# The Polynomial Regression model (degree 2) significantly reduces both the MSE and RMSE compared to the Linear Regression model.
# This improvement suggests that the relationship between the feature 6 and the target is better captured by the polynomial model.
# Comparing performance of the polynomial model and the linear regression model we can see that the output is the same.

# Save the model and the polynomial transformer
joblib.dump(poly_model, 'poly_model.pkl')
joblib.dump(poly, 'poly_transformer.pkl')

