import pandas as pd
import numpy as np
import joblib


# Load the hidden test data
hidden_test_data = pd.read_csv('hidden_test.csv')

# Load the saved model and polynomial features
poly_model = joblib.load('poly_model.pkl')
poly = joblib.load('poly_transformer.pkl')

# Transform feature 6 using logarithms in hidden test data
hidden_test_data['log_feature_6'] = np.log(np.abs(hidden_test_data['6']) + 1)

# Create polynomial features for hidden test data
X_hidden_test = hidden_test_data[['log_feature_6']]
X_hidden_test_poly = poly.transform(X_hidden_test)

# Predict on the hidden test data
hidden_test_pred_log = poly_model.predict(X_hidden_test_poly)

# Transform predictions back to original scale
hidden_test_pred_exp = np.exp(hidden_test_pred_log)

# Save the predictions
hidden_test_data['predicted_target'] = hidden_test_pred_exp
hidden_test_data.to_csv('hidden_test_predictions.csv', index=False)

print("Predictions saved to 'hidden_test_predictions.csv")