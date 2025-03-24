import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from flask import Flask, request, jsonify

# Load dataset
filename = r"C:\Users\rajee\Downloads\pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)

# Prepare data
X = dataframe.iloc[:, 0:8].values
Y = dataframe.iloc[:, 8].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Hyperparameter tuning
clf = SVC()
param_grid = [{'kernel': ['rbf'], 'gamma': [50, 5, 10, 0.5], 'C': [15, 14, 13, 12, 11, 10, 0.1, 0.001]}]
gsv = GridSearchCV(clf, param_grid, cv=10)
gsv.fit(X_train, y_train)
print("Best Parameters:", gsv.best_params_)
print("Best Score:", gsv.best_score_)

# Train best model
clf = SVC(C=gsv.best_params_['C'], gamma=gsv.best_params_['gamma'])
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "model.pkl")

# Flask API setup
app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
