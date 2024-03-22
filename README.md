# Extract features (R&D Spend, Administration, Marketing and Spend) and target variable (Profit)
X = data[['R&D Spend', 'Administration', 'Marketing Spend' , 'State']]
y = data['Profit']

# Perform one-hot encoding for the 'State' column
X_encoded = pd.get_dummies(X, columns=['State'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

Mean Squared Error: 82010363.04
R-squared Score: 0.90

# Visualize the actual vs. predicted values
Actual vs. Predicted Profit using Linear Regression

# Print regression coefficients between -1 and 1
Regression Coefficients between -1 and 1:
R&D Spend: 0.8056
Administration: -0.0688
Marketing Spend: 0.0299

Variance Score (R-squared): 0.8987
Mean Squared Error: 82010363.04

R&D Spend (0.8056):

A one-unit increase in R&D Spend predicts a 0.8056 unit increase in Profit.
Positive correlation: Higher R&D Spend is associated with higher Profit.

Administration (-0.0688):

A one-unit increase in Administration spending predicts a 0.0688 unit decrease in Profit.
Negative correlation: Higher Administration spending is associated with lower Profit.

Marketing Spend (0.0299):

A one-unit increase in Marketing Spend predicts a 0.0299 unit increase in Profit.
Positive correlation: Higher Marketing Spend is associated with higher Profit.

# Plot residual error in training and testing data
Residuals in Training Data

Residuals in Testing Data

Zero Residual Error Line with Adjusted Y-axis Limits

Scatter Points (Blue): Training Data residuals with the predicted values.

Scatter Points (Green): Testing Data residuals with the predicted values.

Horizontal Line (y=0): Represents zero residual error. Points above indicate overestimation, and below indicate underestimation.

Interpretation: Visualizes how well the model's predictions align with the actual values. Adjusting the y-axis limits provides a less magnified view, allowing for better observation of the overall pattern and trends in residuals.

Hypothesis for Linear Regression Model
Null Hypothesis (H0): There is no significant linear relationship between the predictor variables (R&D Spend, Administration, Marketing Spend) and the target variable (Profit).

Alternative Hypothesis (H1): There exists a significant linear relationship between at least one of the predictor variables and the target variable.

Interpretation:

If the p-value associated with the F-statistic is less than the chosen significance level (e.g., 0.05), we reject the null hypothesis.
Rejection of the null hypothesis indicates that the model has statistically significant explanatory power, and at least one predictor variable contributes to predicting the target variable (Profit).
