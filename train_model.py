import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('Student_performance_data _.csv')

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Drop StudentID, Ethnicity, and GradeClass as they are not needed
data = data.drop(columns=['StudentID', 'Ethnicity', 'GradeClass'])

# Separating features (X) and target (y)
X = data.drop(columns=['GPA'])
y = data['GPA']

# Print the shape and columns of X to verify the number of features
print(f"Features used for training: {X.columns.tolist()}")
print(f"Shape of X (features): {X.shape}")

# Scaling the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model and scaler using pickle
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
