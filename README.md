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

  
