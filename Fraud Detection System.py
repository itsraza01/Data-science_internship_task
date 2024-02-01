#Fraud Detection System_task_5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('transactions.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('fraud', axis=1), data['fraud'], test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
