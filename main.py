# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Path to the diamond data file
diamond_data_path = 'Diamonds Prices2022.csv'

# Read in the diamond data using pandas
diamond_data = pd.read_csv(diamond_data_path)

# Drop any rows with missing data
diamond_data = diamond_data.dropna(axis=0)

# One-hot encode the categorical variables (cut, color, clarity)
# using sklearn's OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
cut_color_clarity_encoded = pd.DataFrame(ohe.fit_transform(diamond_data[['cut', 'color', 'clarity']]))
cut_color_clarity_encoded.columns = ohe.get_feature_names_out(['cut', 'color', 'clarity'])
diamond_data = pd.concat([diamond_data, cut_color_clarity_encoded], axis=1)

# Define the features and target variable for the model
diamond_data_features = ['carat', 'depth', 'table', 'x', 'y', 'z'] + list(cut_color_clarity_encoded.columns)
X = diamond_data[diamond_data_features]
y = diamond_data.price

# Split the data into training and validation sets using sklearn's train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define and fit the decision tree regression model
diamond_data_model = DecisionTreeRegressor(random_state=1)
diamond_data_model.fit(train_X, train_y)

# Predict the prices for the validation set and calculate the mean absolute error
pred_val_y = diamond_data_model.predict(val_X)
overall_mae = mean_absolute_error(val_y, pred_val_y)

# Print the overall mean absolute error of the model
print(f'The overall MAE of the model is {overall_mae:.2f}')
