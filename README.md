# Predict wine quality with supervised models

The wine quality dataset (winequality.txt) contains 11 explanatory variables characterizing 1200 wine quality
evaluations. The class informs us whether the wine was qualified as good or not:

# Explanatory variables for the wine quality dataset
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Class membership variable for the wine quality dataset
- Class: good, notgood

Here we try to predict whether a new wine will be qualified as good or not according to the variables above

# Models used and accuracy 

Model Accuracy (rounded down):
- Random Forest 75%
- Logistic Regression 75%
- SVM Linear 72%
- SVM RBF 72%
- KNN 60%

Overall, Random Forest remains the model with the best accuracy on the validation dataset. 
