# Regression-on-the-tabular-data.-General-Machine-Learning
Basic Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 90000 entries, 0 to 89999
Data columns (total 54 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       90000 non-null  int64  
 1   1       90000 non-null  int64  
 2   2       90000 non-null  int64  
 3   3       90000 non-null  int64  
 4   4       90000 non-null  int64  
 5   5       90000 non-null  int64  
 6   6       90000 non-null  float64
 7   7       90000 non-null  float64
 8   8       90000 non-null  int64  
 9   9       90000 non-null  int64  
 10  10      90000 non-null  int64  
 11  11      90000 non-null  int64  
 12  12      90000 non-null  int64  
 13  13      90000 non-null  float64
 14  14      90000 non-null  float64
 15  15      90000 non-null  float64
 16  16      90000 non-null  float64
 17  17      90000 non-null  float64
 18  18      90000 non-null  float64
 19  19      90000 non-null  float64
 20  20      90000 non-null  float64
 21  21      90000 non-null  float64
 22  22      90000 non-null  float64
 23  23      90000 non-null  float64
 24  24      90000 non-null  float64
 25  25      90000 non-null  float64
 26  26      90000 non-null  float64
 27  27      90000 non-null  float64
 28  28      90000 non-null  float64
 29  29      90000 non-null  float64
 30  30      90000 non-null  float64
 31  31      90000 non-null  float64
 32  32      90000 non-null  float64
 33  33      90000 non-null  float64
 34  34      90000 non-null  float64
 35  35      90000 non-null  float64
 36  36      90000 non-null  float64
 37  37      90000 non-null  float64
 38  38      90000 non-null  float64
 39  39      90000 non-null  float64
 40  40      90000 non-null  float64
 41  41      90000 non-null  float64
 42  42      90000 non-null  float64
 43  43      90000 non-null  float64
 44  44      90000 non-null  float64
 45  45      90000 non-null  float64
 46  46      90000 non-null  float64
 47  47      90000 non-null  float64
 48  48      90000 non-null  float64
 49  49      90000 non-null  float64
 50  50      90000 non-null  float64
 51  51      90000 non-null  float64
 52  52      90000 non-null  float64
 53  target  90000 non-null  float64
dtypes: float64(43), int64(11)
memory usage: 37.1 MB
None


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
